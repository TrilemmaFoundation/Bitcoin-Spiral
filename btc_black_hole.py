# ------------------------------------------------------------
# Necessary Imports
# ------------------------------------------------------------
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation, PillowWriter

import time
from requests.exceptions import HTTPError
# If you have coinmetrics installed:
from coinmetrics.api_client import CoinMetricsClient


# ------------------------------------------------------------
# Optional: Disable TeX rendering
# ------------------------------------------------------------
mpl.rcParams['text.usetex'] = False

# ------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ------------------------------------------------------------
# Initialize Coin Metrics API client
# ------------------------------------------------------------
client = CoinMetricsClient()  # Adjust if you have your own credentials


# ------------------------------------------------------------
# Define asset, metric, and time range
# ------------------------------------------------------------
asset = 'btc'
metric = 'PriceUSD'
start_time = '2010-01-01'
end_time = datetime.today().strftime('%Y-%m-%d')
frequency = '1d'

logging.info("Fetching BTC PriceUSD (to invert to USD/BTC)...")
try:
    df_raw = client.get_asset_metrics(
        assets=asset,
        metrics=[metric],
        frequency=frequency,
        start_time=start_time,
        end_time=end_time
    ).to_dataframe()
except HTTPError as e:
    logging.error(f"Error fetching data: {e}")
    raise


# ------------------------------------------------------------
# Prepare DataFrame
# ------------------------------------------------------------
df_raw = df_raw.rename(columns={metric: 'BTC_in_USD'})
df_raw['time'] = pd.to_datetime(df_raw['time']).dt.normalize().dt.tz_localize(None)
df_raw.set_index('time', inplace=True)
df_raw.sort_index(inplace=True)

# Create the inverted price: USD/BTC
df_raw['USD_per_BTC'] = 1.0 / df_raw['BTC_in_USD']


# ------------------------------------------------------------
# Define Halving Dates
# ------------------------------------------------------------
halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20')
]


# ------------------------------------------------------------
# Identify All-Time Lows & Cycle Lows (in the inverted price)
# ------------------------------------------------------------
all_time_lows = df_raw[df_raw['USD_per_BTC'] == df_raw['USD_per_BTC'].cummin()]

cycle_lows = df_raw.loc[df_raw.index.isin([
    df_raw.loc[halving_dates[i-1]:halving_dates[i]]['USD_per_BTC'].idxmin()
    for i in range(1, len(halving_dates))
    if not df_raw.loc[halving_dates[i-1]:halving_dates[i]].empty
])]


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_price_4_years_prior(current_date, df, col='USD_per_BTC'):
    date_4y_prior = current_date - pd.DateOffset(years=4)
    if date_4y_prior < df.index[0]:
        return None
    pos = df.index.get_indexer([date_4y_prior], method='nearest')[0]
    return df[col].iloc[pos]


def _calculate_theta(date, fixed_halving):
    """
    1458 days ~ 2*pi (one full circle).
    Negative sign for forward rotation.
    """
    days_since_fixed = (date - fixed_halving).days
    return (-2 * np.pi * days_since_fixed / 1458) + (np.pi / 2)


def _set_theta_labels(ax):
    """
    Label halving years around the circle at 0, 90, 180, 270 degrees.
    Adjust label positions to move them closer to the plot.
    """
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=[
            [2009, 2013, 2017, 2021, 2025],   # 0°
            [2012, 2016, 2020, 2024, 2028],  # 90°
            [2011, 2015, 2019, 2023, 2027],  # 180°
            [2010, 2014, 2018, 2022, 2026]   # 270°
        ],
        fontsize=14,
        color='white'
    )
    # Tweak label positions
    labels = ax.get_xticklabels()
    # 0°
    labels[0].set_y(-0.15)
    # 90°
    labels[1].set_y(-0.08)
    # 180°
    labels[2].set_y(-0.15)
    # 270°
    labels[3].set_y(-0.08)


# ------------------------------------------------------------
# Main Animation
# ------------------------------------------------------------
def animate_inward_spiral(df, duration=10, fps=15, pause_duration=2):
    """
    Creates an inward spiral of USD/BTC from 100 BTC down to 0 in radial ticks.
    """

    total_frames = duration * fps
    pause_frames = pause_duration * fps

    atl_idx = df.index.get_indexer(all_time_lows.index, method='nearest')
    cyc_low_idx = df.index.get_indexer(cycle_lows.index, method='nearest')
    halv_idx = df.index.get_indexer(halving_dates, method='nearest')

    marker_indices = np.unique(np.concatenate([atl_idx, cyc_low_idx, halv_idx]))

    linear_indices = np.linspace(0, len(df) - 1, total_frames, dtype=int)
    indices = np.unique(np.concatenate([linear_indices, marker_indices]))
    df = df.iloc[indices].copy()

    # Radius example
    BASELINE_BTC = 100.0
    def compute_radius(usd_btc):
        with np.errstate(divide='ignore'):
            r_val = np.log10(BASELINE_BTC / usd_btc)
        return np.clip(r_val, -10, 8)

    df['r'] = df['USD_per_BTC'].apply(compute_radius)

    fixed_halving = pd.Timestamp('2024-04-20')
    df['theta'] = df.index.to_series().apply(lambda d: _calculate_theta(d, fixed_halving))

    fig = plt.figure(figsize=(14, 10), facecolor='black')

    # Adjust top spacing so the title is closer
    # (You can adjust top=0.88 or so if you need even less space.)
    plt.subplots_adjust(top=0.92)

    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('black')

    # Adjust the pad on the title to bring it closer
    ax.set_title('USD/BTC Inward Spiral', fontsize=24, color='white', pad=15)

    ax.grid(True, color='gray', linestyle='--')
    _set_theta_labels(ax)

    tick_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    tick_labels = [
        "100 BTC", "10 BTC", "1 BTC", "0.1 BTC", "0.01 BTC",
        "0.001 BTC", "0.0001 BTC", "0.00001 BTC", "0"
    ]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, color='white')
    ax.set_ylim(0, 8)
    ax.invert_yaxis()

    line, = ax.plot([], [], lw=1.5, color='lightblue')

    # Halving
    halving_markers = []
    halving_labels = [
        '1st Halving (2012-11-28)',
        '2nd Halving (2016-07-09)',
        '3rd Halving (2020-05-11)',
        '4th Halving (2024-04-20)'
    ]
    for i in range(len(halving_dates)):
        (marker,) = ax.plot([], [], 'ws', markersize=8, label=halving_labels[i])
        halving_markers.append(marker)

    # All-Time Low
    atl_markers, = ax.plot([], [], 'bo', markersize=8, markerfacecolor='none', label='All-Time Low')
    # Cycle Low
    cyc_low_markers, = ax.plot([], [], 'ro', markersize=8, label='Cycle Low')

    fig.legend(
        loc='upper left', bbox_to_anchor=(0, 1),
        fontsize=12, facecolor='black', labelcolor='white'
    )

    text_box = ax.text(
        0.5, 0.02, '',
        transform=ax.transAxes,
        fontsize=14, color='white',
        verticalalignment='bottom', horizontalalignment='center',
        bbox=dict(facecolor='black', alpha=0.7, pad=6)
    )

    def init():
        line.set_data([], [])
        for mk in halving_markers:
            mk.set_data([], [])
        atl_markers.set_data([], [])
        cyc_low_markers.set_data([], [])
        text_box.set_text('')
        return [line, *halving_markers, atl_markers, cyc_low_markers, text_box]

    def update(frame_idx):
        max_i = min(frame_idx, len(df) - 1)
        line.set_data(df['theta'][:max_i], df['r'][:max_i])

        current_time = df.index[max_i]
        current_date_str = current_time.strftime('%Y-%m-%d')
        current_price = df['USD_per_BTC'].iloc[max_i]

        prior_4y = get_price_4_years_prior(current_time, df_raw, col='USD_per_BTC')
        if prior_4y is None:
            txt = f"Date: {current_date_str}\nUSD/BTC: {current_price:.8f} BTC"
        else:
            txt = (
                f"Date: {current_date_str}\n"
                f"USD/BTC: {current_price:.8f} BTC > {prior_4y:.8f} (4y ago)"
            )
        text_box.set_text(txt)

        # Halving markers
        for i, h_date in enumerate(halving_dates):
            if current_time >= h_date and h_date in df.index:
                h_r = df.loc[h_date, 'r']
                h_th = df.loc[h_date, 'theta']
                halving_markers[i].set_data([h_th], [h_r])
            else:
                halving_markers[i].set_data([], [])

        # All-Time Low
        atl_filtered = df.index.intersection(all_time_lows.index)
        atl_filtered = atl_filtered[atl_filtered <= current_time]
        if len(atl_filtered) > 0:
            atl_r = df.loc[atl_filtered, 'r']
            atl_t = df.loc[atl_filtered, 'theta']
            atl_markers.set_data(atl_t, atl_r)

        # Cycle Low
        cl_filtered = df.index.intersection(cycle_lows.index)
        cl_filtered = cl_filtered[cl_filtered <= current_time]
        if len(cl_filtered) > 0:
            cl_r = df.loc[cl_filtered, 'r']
            cl_t = df.loc[cl_filtered, 'theta']
            cyc_low_markers.set_data(cl_t, cl_r)

        return [line, *halving_markers, atl_markers, cyc_low_markers, text_box]

    ani = FuncAnimation(
        fig, update,
        frames=len(df) + pause_frames,
        init_func=init,
        blit=True,
        interval=1000/fps,
        repeat=True
    )

    outdir = "gifs"
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(
        outdir,
        "usd_btc_inward_spiral_" + datetime.now().strftime("%b_%d_%H_%M").lower() + ".gif"
    )
    ani.save(fname, writer=PillowWriter(fps=fps), dpi=72)
    logging.info(f"Inward spiral animation saved to {fname}")

    plt.show()


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------
if __name__ == "__main__":
    animate_inward_spiral(df_raw, duration=10, fps=15, pause_duration=2)