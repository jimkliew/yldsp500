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

START_DATE = "1982-01-01"
TIMEZONE = "America/New_York"
END_DATE = pd.Timestamp.now("UTC").tz_convert(TIMEZONE).normalize().strftime("%Y-%m-%d")
BASE_FRAME_STEP = 44
INVERSION_DETAIL_STEP = 5
BASE_FRAME_MS = 70
FOCUS_FRAME_MS = 180
FINAL_FRAME_MS = 700
MP4_FPS = 14
QUARTER_FORWARD_DAYS = 63
MIN_INVERSION_DAYS = 20


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


def build_dataset(refresh: bool = False) -> pd.DataFrame:
    yields_daily = [fetch_fred_series(spec.code, refresh=refresh) for spec in YIELD_SERIES]
    sp500_daily = fetch_stooq_sp500(refresh=refresh)
    recession_monthly = fetch_fred_series(RECESSION_SERIES.code, refresh=refresh)

    yields = pd.concat(yields_daily, axis=1, sort=True).loc[START_DATE:END_DATE]
    sp500 = sp500_daily.loc[START_DATE:END_DATE]

    dataset = yields.join(sp500.rename("SP500"), how="inner")
    dataset = dataset.sort_index().ffill()
    dataset = dataset.dropna(subset=["SP500", "DGS3MO", "DGS10"])
    dataset["curve_spread"] = dataset["DGS10"] - dataset["DGS3MO"]
    dataset["sp500_indexed"] = dataset["SP500"] / dataset["SP500"].iloc[0] * 100
    dataset["sp500_forward_quarter_pct"] = dataset["SP500"].shift(-QUARTER_FORWARD_DAYS) / dataset["SP500"] - 1

    recession_daily = recession_monthly.resample("D").ffill().loc[dataset.index.min() : dataset.index.max()]
    recession_daily = recession_daily.reindex(dataset.index, method="ffill").fillna(0)
    recession_daily.name = "USREC"
    dataset = dataset.join(recession_daily, how="left")
    dataset["USREC"] = dataset["USREC"].ffill().fillna(0)
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

    detail_dates: set[pd.Timestamp] = set()
    inversion_mask = sustained_inversion_mask(data)
    focus_mask = pd.Series(False, index=data.index)
    starts = inversion_starts(data)

    for start in starts:
        pos = int(data.index.get_loc(start))
        lo = max(0, pos - 22)
        hi = min(len(data.index), pos + 44)
        focus_mask.iloc[lo:hi] = True
        detail_dates.update(data.index[lo:hi:INVERSION_DETAIL_STEP].tolist())

    focus_mask |= inversion_mask
    all_dates = sorted(set(base_dates).union(detail_dates))
    dates = pd.DatetimeIndex(all_dates)
    durations = [FOCUS_FRAME_MS if bool(focus_mask.loc[date]) else BASE_FRAME_MS for date in dates]
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
    curve_color = "#b4233c" if is_inverted else "#0a6c74"
    fill_color = "#f5b3bd" if is_inverted else "#b9e0da"
    panel_face = "#fff7f8" if is_inverted else "#fffdf7"

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#f8f6ef")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.45], hspace=0.24)
    ax_curve = fig.add_subplot(gs[0])
    ax_sp = fig.add_subplot(gs[1])

    ax_curve.set_facecolor(panel_face)
    ax_curve.grid(color="#d9d2c1", linewidth=0.8, alpha=0.6)
    ax_curve.plot(maturities, curve, color=curve_color, marker="o", linewidth=3.6, markersize=7, zorder=3)
    ax_curve.fill_between(maturities, curve, 0, color=fill_color, alpha=0.45, zorder=2)
    ax_curve.axhline(0, color="#867a64", linewidth=1.2, zorder=1)
    ax_curve.set_xlim(0, 31)
    ax_curve.set_ylim(min(-2.0, data[[spec.code for spec in YIELD_SERIES]].min().min() - 0.6), max(16, data[[spec.code for spec in YIELD_SERIES]].max().max() + 0.6))
    ax_curve.set_xticks(maturities)
    ax_curve.set_xticklabels(maturity_labels, fontsize=10)
    ax_curve.set_ylabel("Yield (%)", fontsize=11)
    ax_curve.set_title("US Treasury Yield Curve", loc="left", fontsize=20, fontweight="bold", color="#1d1a14", pad=12)
    ax_curve.text(
        0.995,
        1.04,
        frame_date.strftime("%b %d, %Y"),
        transform=ax_curve.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        color="#1d1a14",
        fontweight="bold",
    )

    spread_label = "INVERTED" if is_inverted else "Normal"
    ax_curve.text(
        0.02,
        0.86,
        f"10Y - 3M spread: {spread:+.2f} pts  |  {spread_label}",
        transform=ax_curve.transAxes,
        fontsize=12,
        color=curve_color,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff9ea", "edgecolor": curve_color, "linewidth": 1.3},
        zorder=5,
    )

    if is_inverted:
        ax_curve.text(
            0.98,
            0.17,
            "INVERSION",
            transform=ax_curve.transAxes,
            ha="right",
            va="center",
            fontsize=20,
            color="#b4233c",
            fontweight="bold",
            alpha=0.9,
        )

    tracker = ax_curve.inset_axes([0.62, 0.12, 0.34, 0.28])
    tracker.set_facecolor("#fffaf3")
    tracker.plot(data.index, data["curve_spread"], color="#7c7462", linewidth=1.2)
    tracker.fill_between(
        data.index,
        data["curve_spread"],
        0,
        where=data["curve_spread"] < 0,
        color="#d85169",
        alpha=0.35,
    )
    tracker.axhline(0, color="#867a64", linewidth=1.0)
    tracker.axvline(frame_date, color=curve_color, linewidth=1.3, linestyle="--")
    tracker.scatter([frame_date], [spread], color=curve_color, s=28, zorder=4)
    tracker.set_title("10Y - 3M Tracker", fontsize=8.5, loc="left", pad=2)
    tracker.set_xlim(data.index.min(), data.index.max())
    tracker.set_ylim(data["curve_spread"].min() - 0.8, data["curve_spread"].max() + 0.8)
    tracker.xaxis.set_major_locator(mdates.YearLocator(base=10))
    tracker.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tracker.tick_params(axis="x", labelsize=6, pad=1)
    tracker.tick_params(axis="y", labelsize=6, pad=1)
    for spine in tracker.spines.values():
        spine.set_color("#867a64")

    ax_sp.set_facecolor("#fffdf7")
    for start, end in recessions:
        ax_sp.axvspan(start, end, color="#c8c3b3", alpha=0.32, linewidth=0, zorder=0)
    for start, end in inversion_periods:
        ax_sp.axvspan(start, end, color="#ef8e9d", alpha=0.16, linewidth=0, zorder=1)

    ax_sp.plot(data.index, data["sp500_indexed"], color="#d3cec0", linewidth=2.0, alpha=0.7, zorder=2)
    ax_sp.plot(subset.index, subset["sp500_indexed"], color="#1b4d9b", linewidth=3.0, zorder=4)
    ax_sp.axvline(frame_date, color="#111111", linewidth=1.4, linestyle="--", zorder=5)
    ax_sp.scatter([frame_date], [current["sp500_indexed"]], s=50, color=curve_color, edgecolor="#fffdf7", linewidth=0.9, zorder=6)

    seen_inversions = inversion_points[inversion_points <= frame_date]
    if len(seen_inversions) > 0:
        ax_sp.scatter(
            seen_inversions,
            data.loc[seen_inversions, "sp500_indexed"],
            color="#b4233c",
            edgecolor="#fffdf7",
            linewidth=0.7,
            s=34,
            zorder=6,
        )

    ax_sp.set_title("S&P 500 Price Index With Recessions and Inversions", loc="left", fontsize=20, fontweight="bold", color="#1d1a14", pad=10)
    ax_sp.set_ylabel("Indexed to 100 (log scale)", fontsize=11)
    ax_sp.set_xlim(data.index.min(), data.index.max())
    ax_sp.set_ylim(data["sp500_indexed"].min() * 0.9, data["sp500_indexed"].max() * 1.08)
    ax_sp.set_yscale("log")
    ax_sp.grid(color="#d9d2c1", linewidth=0.8, alpha=0.6)
    ax_sp.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_sp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_sp.tick_params(axis="x", labelrotation=0)

    note_lines = [
        f"S&P 500 level: {current['SP500']:.0f}",
        f"Next ~quarter move from here: {format_pct(current['sp500_forward_quarter_pct'])}",
        "Grey bands: NBER recessions",
        "Pink bands: active inversion periods",
        "Red dots: first day of inversion",
    ]
    ax_sp.text(
        0.015,
        0.975,
        "\n".join(note_lines),
        transform=ax_sp.transAxes,
        va="top",
        fontsize=11,
        color="#1d1a14",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff9ea", "edgecolor": "#c8b991", "linewidth": 1.0},
        zorder=7,
    )

    source_text = (
        "Sources: FRED Treasury/NBER recession series and Stooq daily S&P 500 history"
        " | S&P 500 shown as price index, not total return"
    )
    fig.text(0.015, 0.015, source_text, fontsize=9.6, color="#5b5446")

    for axis in (ax_curve, ax_sp):
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
    palette_frames = [frame.convert("P", palette=Image.ADAPTIVE, colors=160) for frame in frames]
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
            "scale=1280:720:flags=lanczos",
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
    gif_path, mp4_path = build_assets(refresh=True)
    print(gif_path)
    print(mp4_path)
