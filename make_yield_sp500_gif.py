#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
GIF_PATH = OUTPUT_DIR / "yield_curve_sp500_recessions.gif"
MP4_PATH = OUTPUT_DIR / "yield_curve_sp500_recessions.mp4"
FRAME_WIDTH = 1680
FRAME_HEIGHT = 900

START_DATE = "1982-01-01"
TIMEZONE = "America/New_York"
END_DATE = pd.Timestamp.now("UTC").tz_convert(TIMEZONE).normalize().strftime("%Y-%m-%d")
BASE_FRAME_STEP = 44
BASE_FRAME_MS = 85
FINAL_FRAME_MS = 700
MP4_FPS = 14
QUARTER_FORWARD_DAYS = 63
MIN_INVERSION_DAYS = 20
SPREAD_SHORT_CODE = "DGS2"
SPREAD_LONG_CODE = "DGS10"
SPREAD_LABEL = "10Y - 2Y"
CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.titleweight"] = "normal"
matplotlib.rcParams["font.weight"] = "regular"


@dataclass(frozen=True)
class SeriesSpec:
    code: str
    label: str
    years: float
    source: str


YIELD_SERIES = [
    SeriesSpec("DGS3MO", "3M", 0.25, "Treasury constant maturity"),
    SeriesSpec("DGS6MO", "6M", 0.50, "Treasury constant maturity"),
    SeriesSpec("DGS1", "1Y", 1.0, "Treasury constant maturity"),
    SeriesSpec("DGS2", "2Y", 2.0, "Treasury constant maturity"),
    SeriesSpec("DGS3", "3Y", 3.0, "Treasury constant maturity"),
    SeriesSpec("DGS5", "5Y", 5.0, "Treasury constant maturity"),
    SeriesSpec("DGS7", "7Y", 7.0, "Treasury constant maturity"),
    SeriesSpec("DGS10", "10Y", 10.0, "Treasury constant maturity"),
    SeriesSpec("DGS20", "20Y", 20.0, "Treasury constant maturity"),
    SeriesSpec("DGS30", "30Y", 30.0, "Treasury constant maturity"),
]
RECESSION_SERIES = SeriesSpec("USREC", "US Recession Indicator", 0.0, "NBER via FRED")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def fred_csv_url(series_id: str) -> str:
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    )


def fetch_fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    cache_path = DATA_DIR / f"{series_id}.csv"
    if refresh or not cache_path.exists():
        response = requests.get(fred_csv_url(series_id), timeout=30)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")

    frame = pd.read_csv(cache_path)
    date_col = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_col not in frame.columns:
        raise ValueError(f"Unexpected FRED CSV columns for {series_id}: {frame.columns.tolist()}")

    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.rename(columns={date_col: "DATE"})
    values = pd.to_numeric(frame[series_id].replace(".", np.nan), errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame["DATE"]), name=series_id).sort_index()


def fetch_stooq_sp500(refresh: bool = False) -> pd.Series:
    cache_path = DATA_DIR / "SP500_STOOQ_DAILY.csv"
    if refresh or not cache_path.exists():
        response = requests.get(
            "https://stooq.com/q/d/l/?s=%5Espx&i=d",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")

    frame = pd.read_csv(cache_path, parse_dates=["Date"])
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.rename(columns={"Date": "DATE"})
    return pd.Series(frame["Close"].to_numpy(), index=pd.DatetimeIndex(frame["DATE"]), name="SP500").sort_index()


def fetch_multpl_cape(refresh: bool = False) -> pd.Series:
    cache_path = DATA_DIR / "SP500_CAPE_MULTPL.csv"
    if refresh or not cache_path.exists():
        tables = pd.read_html(CAPE_URL)
        if not tables:
            raise ValueError("No tables found at Multpl CAPE source")
        tables[0].to_csv(cache_path, index=False)

    frame = pd.read_csv(cache_path)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "Date" not in frame.columns or "Value" not in frame.columns:
        raise ValueError(f"Unexpected Multpl CAPE columns: {frame.columns.tolist()}")

    frame["DATE"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = (
        frame["Value"]
        .astype(str)
        .str.replace(r"[^0-9.\\-]", "", regex=True)
        .replace("", np.nan)
    )
    frame["CAPE_RATIO"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["DATE", "CAPE_RATIO"])
    series = pd.Series(frame["CAPE_RATIO"].to_numpy(), index=pd.DatetimeIndex(frame["DATE"]), name="CAPE_RATIO")
    return series.sort_index()


def build_dataset(refresh: bool = False) -> pd.DataFrame:
    yields_daily = [fetch_fred_series(spec.code, refresh=refresh) for spec in YIELD_SERIES]
    sp500_daily = fetch_stooq_sp500(refresh=refresh)
    cape_monthly = fetch_multpl_cape(refresh=refresh)
    recession_monthly = fetch_fred_series(RECESSION_SERIES.code, refresh=refresh)

    yields = pd.concat(yields_daily, axis=1, sort=True).loc[START_DATE:END_DATE]
    sp500 = sp500_daily.loc[START_DATE:END_DATE]

    dataset = yields.join(sp500.rename("SP500"), how="inner")
    dataset = dataset.sort_index()
    dataset = dataset.dropna(subset=["SP500", SPREAD_SHORT_CODE, SPREAD_LONG_CODE])
    dataset["curve_spread"] = dataset[SPREAD_LONG_CODE] - dataset[SPREAD_SHORT_CODE]
    dataset["sp500_indexed"] = dataset["SP500"] / dataset["SP500"].iloc[0] * 100
    dataset["sp500_forward_quarter_pct"] = dataset["SP500"].shift(-QUARTER_FORWARD_DAYS) / dataset["SP500"] - 1
    cape_daily = cape_monthly.resample("D").ffill().loc[dataset.index.min() : dataset.index.max()]
    dataset = dataset.join(cape_daily.reindex(dataset.index, method="ffill"), how="left")

    recession_daily = recession_monthly.resample("D").ffill().loc[dataset.index.min() : dataset.index.max()]
    recession_daily = recession_daily.reindex(dataset.index, method="ffill").fillna(0)
    recession_daily.name = "USREC"
    dataset = dataset.join(recession_daily, how="left")
    dataset["USREC"] = dataset["USREC"].ffill().fillna(0)
    dataset["CAPE_RATIO"] = dataset["CAPE_RATIO"].ffill()
    return dataset


def contiguous_windows(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = mask.astype(int)
    starts = values[(values == 1) & (values.shift(fill_value=0) == 0)].index
    ends = values[(values == 1) & (values.shift(-1, fill_value=0) == 0)].index
    return list(zip(starts, ends, strict=True))


def recession_windows(recession_series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return contiguous_windows(recession_series > 0)


def sustained_inversion_mask(data: pd.DataFrame) -> pd.Series:
    raw = data["curve_spread"] < 0
    mask = pd.Series(False, index=data.index)
    for start, end in contiguous_windows(raw):
        days = int(data.index.get_loc(end)) - int(data.index.get_loc(start)) + 1
        if days >= MIN_INVERSION_DAYS:
            mask.loc[start:end] = True
    return mask


def inversion_starts(data: pd.DataFrame) -> pd.DatetimeIndex:
    inverted = sustained_inversion_mask(data)
    return data.index[inverted & (~inverted.shift(fill_value=False))]


def inversion_windows(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return contiguous_windows(sustained_inversion_mask(data))


def build_frame_schedule(data: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[int]]:
    base_dates = list(data.index[::BASE_FRAME_STEP])
    if not base_dates or base_dates[-1] != data.index[-1]:
        base_dates.append(data.index[-1])
    dates = pd.DatetimeIndex(base_dates)
    durations = [BASE_FRAME_MS for _ in dates]
    durations[-1] = FINAL_FRAME_MS
    return dates, durations


def format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:+.1f}%"


def curve_points(current: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    points = [
        (spec.years, float(current[spec.code]), spec.label)
        for spec in YIELD_SERIES
        if spec.code in current.index and pd.notna(current[spec.code])
    ]
    maturities = np.array([point[0] for point in points], dtype=float)
    yields = np.array([point[1] for point in points], dtype=float)
    labels = [point[2] for point in points]
    return maturities, yields, labels


def render_frame(
    data: pd.DataFrame,
    frame_date: pd.Timestamp,
    recessions: list[tuple[pd.Timestamp, pd.Timestamp]],
    inversion_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    inversion_points: pd.DatetimeIndex,
) -> Image.Image:
    subset = data.loc[:frame_date].copy()
    current = subset.iloc[-1]
    maturities, curve, maturity_labels = curve_points(current)
    spread = float(current["curve_spread"])
    is_inverted = spread < 0
    curve_color = "#b4233c" if is_inverted else "#146c74"
    fill_color = "#f5b3bd" if is_inverted else "#b9e0da"
    panel_face = "#fff7f8" if is_inverted else "#fffdf7"
    short_year = next(spec.years for spec in YIELD_SERIES if spec.code == SPREAD_SHORT_CODE)
    long_year = next(spec.years for spec in YIELD_SERIES if spec.code == SPREAD_LONG_CODE)
    short_yield = float(current[SPREAD_SHORT_CODE])
    long_yield = float(current[SPREAD_LONG_CODE])

    fig = plt.figure(figsize=(16.8, 9.0), dpi=100)
    fig.patch.set_facecolor("#f6f1e8")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 0.78], width_ratios=[0.95, 1.25], hspace=0.22, wspace=0.14)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_pe = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "US Yield Curve, S&P 500, and Valuation",
        x=0.015,
        y=0.972,
        ha="left",
        fontsize=22,
        color="#1d1a14",
        fontfamily="DejaVu Serif",
        fontweight="normal",
    )
    fig.text(
        0.985,
        0.928,
        frame_date.strftime("%b %d, %Y"),
        ha="right",
        fontsize=14,
        color="#1d1a14",
        fontweight="normal",
    )

    ax_curve.set_facecolor("#fffdf9")
    ax_curve.grid(axis="y", color="#ddd4c5", linewidth=0.8, alpha=0.8)
    ax_curve.grid(axis="x", visible=False)
    ax_curve.plot(maturities, curve, color="#1a1a1a", linewidth=2.4, zorder=3)
    ax_curve.scatter(maturities, curve, color="#1a1a1a", s=24, zorder=4)
    ax_curve.plot([short_year, long_year], [short_yield, long_yield], color="#c94b5c", linewidth=2.0, zorder=5)
    ax_curve.scatter([short_year, long_year], [short_yield, long_yield], color="#c94b5c", s=56, edgecolor="#fffdf9", linewidth=0.8, zorder=6)
    ax_curve.axhline(0, color="#9a8f7a", linewidth=1.0, zorder=1)
    ax_curve.set_xlim(0, 31)
    ax_curve.set_ylim(min(-3.0, data[[spec.code for spec in YIELD_SERIES]].min().min() - 1.2), max(17, data[[spec.code for spec in YIELD_SERIES]].max().max() + 1.2))
    ax_curve.set_xticks(maturities)
    ax_curve.set_xticklabels(maturity_labels, fontsize=11)
    ax_curve.tick_params(axis="y", labelsize=11)
    ax_curve.set_ylabel("Yield (%)", fontsize=12)
    ax_curve.set_title("US Treasury Yield Curve", loc="left", fontsize=18, color="#1d1a14", pad=10, fontfamily="DejaVu Serif")

    spread_label = "INVERTED" if is_inverted else "Normal"
    ax_curve.text(
        0.02,
        0.93,
        f"{SPREAD_LABEL} spread {spread:+.2f} pts",
        transform=ax_curve.transAxes,
        fontsize=11.5,
        color="#171717" if not is_inverted else "#b4233c",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbf6eb", "edgecolor": "#d8cfbf", "linewidth": 0.9},
        zorder=5,
    )
    ax_curve.text(
        0.02,
        0.865,
        f"2Y {short_yield:.2f}%   10Y {long_yield:.2f}%   State {spread_label}",
        transform=ax_curve.transAxes,
        fontsize=10.8,
        color="#6b5f4c",
        zorder=5,
    )

    if is_inverted:
        ax_curve.text(
            0.98,
            0.10,
            "INVERSION",
            transform=ax_curve.transAxes,
            ha="right",
            va="center",
            fontsize=18,
            color="#c94b5c",
            alpha=0.9,
        )

    ax_sp.set_facecolor("#fffdf9")
    for start, end in recessions:
        ax_sp.axvspan(start, end, color="#d9d3c5", alpha=0.45, linewidth=0, zorder=0)
    for start, end in inversion_periods:
        ax_sp.axvspan(start, end, color="#f3ccd2", alpha=0.35, linewidth=0, zorder=1)

    ax_sp.plot(data.index, data["sp500_indexed"], color="#d8d1c4", linewidth=1.8, alpha=0.7, zorder=2)
    ax_sp.plot(subset.index, subset["sp500_indexed"], color="#295f9e", linewidth=2.5, zorder=4)
    ax_sp.axvline(frame_date, color="#767676", linewidth=1.0, linestyle="--", zorder=5)
    ax_sp.scatter([frame_date], [current["sp500_indexed"]], s=34, color="#295f9e", edgecolor="#fffdf9", linewidth=0.8, zorder=6)

    seen_inversions = inversion_points[inversion_points <= frame_date]
    if len(seen_inversions) > 0:
        ax_sp.scatter(
            seen_inversions,
            data.loc[seen_inversions, "sp500_indexed"],
            color="#c94b5c",
            edgecolor="#fffdf9",
            linewidth=0.6,
            s=22,
            zorder=5,
        )

    ax_sp.set_title("S&P 500 Price Index", loc="left", fontsize=18, color="#1d1a14", pad=10, fontfamily="DejaVu Serif")
    ax_sp.set_ylabel("Indexed to 100", fontsize=12)
    ax_sp.set_xlim(data.index.min(), data.index.max())
    ax_sp.set_ylim(data["sp500_indexed"].min() * 0.78, data["sp500_indexed"].max() * 1.18)
    ax_sp.set_yscale("log")
    ax_sp.grid(color="#ddd4c5", linewidth=0.8, alpha=0.75)
    ax_sp.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_sp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_sp.tick_params(axis="x", labelrotation=0, labelsize=11)
    ax_sp.tick_params(axis="y", labelsize=11)
    ax_sp.text(
        0.015,
        0.95,
        f"Level {current['SP500']:.0f}\nQuarter ahead {format_pct(current['sp500_forward_quarter_pct'])}",
        transform=ax_sp.transAxes,
        va="top",
        fontsize=11,
        color="#1d1a14",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#fbf6eb", "edgecolor": "#d8cfbf", "linewidth": 0.9},
        zorder=6,
    )

    ax_pe.set_facecolor("#fffdf9")
    for start, end in recessions:
        ax_pe.axvspan(start, end, color="#d9d3c5", alpha=0.45, linewidth=0, zorder=0)
    ax_pe.plot(data.index, data["CAPE_RATIO"], color="#d8d1c4", linewidth=1.6, alpha=0.75, zorder=1)
    ax_pe.plot(subset.index, subset["CAPE_RATIO"], color="#8a6232", linewidth=2.3, zorder=3)
    ax_pe.axvline(frame_date, color="#767676", linewidth=1.0, linestyle="--", zorder=4)
    ax_pe.scatter([frame_date], [current["CAPE_RATIO"]], s=34, color="#8a6232", edgecolor="#fffdf9", linewidth=0.8, zorder=5)
    ax_pe.set_title("S&P 500 Shiller CAPE", loc="left", fontsize=17, color="#1d1a14", pad=8, fontfamily="DejaVu Serif")
    ax_pe.set_ylabel("CAPE", fontsize=12)
    ax_pe.set_xlim(data.index.min(), data.index.max())
    ax_pe.set_ylim(max(3, data["CAPE_RATIO"].min() * 0.75), min(60, data["CAPE_RATIO"].max() * 1.15))
    ax_pe.grid(color="#ddd4c5", linewidth=0.8, alpha=0.75)
    ax_pe.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_pe.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_pe.tick_params(axis="x", labelsize=11)
    ax_pe.tick_params(axis="y", labelsize=11)
    ax_pe.text(
        0.015,
        0.93,
        f"CAPE {current['CAPE_RATIO']:.1f}   Monthly series carried forward daily",
        transform=ax_pe.transAxes,
        va="top",
        fontsize=11,
        color="#1d1a14",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#fbf6eb", "edgecolor": "#d8cfbf", "linewidth": 0.9},
        zorder=6,
    )

    source_text = (
        "Sources: FRED Treasury and NBER recession series, Stooq daily S&P 500 history, and Multpl monthly Shiller CAPE"
        f" | Inversion measure: {SPREAD_LABEL}"
        " | S&P 500 shown as price index, not total return"
    )
    fig.text(0.015, 0.015, source_text, fontsize=9.8, color="#62594c")

    for axis in (ax_curve, ax_sp, ax_pe):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#867a64")
        axis.spines["bottom"].set_color("#867a64")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba[:, :, :3])
    plt.close(fig)
    return image


def build_animation(refresh: bool = False) -> tuple[list[Image.Image], list[int]]:
    ensure_dirs()
    data = build_dataset(refresh=refresh)
    recessions = recession_windows(data["USREC"])
    inversion_periods = inversion_windows(data)
    inversion_points = inversion_starts(data)
    dates, durations = build_frame_schedule(data)
    frames = [render_frame(data, frame_date, recessions, inversion_periods, inversion_points) for frame_date in dates]
    return frames, durations


def save_gif(frames: list[Image.Image], durations: list[int]) -> Path:
    palette_frames = [frame.convert("P", palette=Image.ADAPTIVE, colors=192) for frame in frames]
    palette_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return GIF_PATH


def save_mp4(frames: list[Image.Image], durations: list[int]) -> Path:
    with tempfile.TemporaryDirectory(prefix="yield-sp500-frames-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        repeat_index = 0
        for frame, duration in zip(frames, durations, strict=True):
            repeat_count = max(1, round(duration / (1000 / MP4_FPS)))
            rgb_frame = frame.convert("RGB")
            for _ in range(repeat_count):
                rgb_frame.save(tmp_path / f"frame_{repeat_index:05d}.png")
                repeat_index += 1

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(MP4_FPS),
            "-i",
            str(tmp_path / "frame_%05d.png"),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=lanczos",
            str(MP4_PATH),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return MP4_PATH


def build_assets(refresh: bool = False) -> tuple[Path, Path]:
    frames, durations = build_animation(refresh=refresh)
    gif_path = save_gif(frames, durations)
    mp4_path = save_mp4(frames, durations)
    return gif_path, mp4_path


if __name__ == "__main__":
    gif_path, mp4_path = build_assets(refresh=False)
    print(gif_path)
    print(mp4_path)
