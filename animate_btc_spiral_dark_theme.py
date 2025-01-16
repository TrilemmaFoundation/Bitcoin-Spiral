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
# Optional: Disable all TeX rendering (can also help if $ is still misread)
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
client = CoinMetricsClient()  # Adjust if you have your own API credentials


# ------------------------------------------------------------
# Define asset, metric, and time range
# ------------------------------------------------------------
asset = 'btc'
metric = 'PriceUSD'
start_time = '2010-01-01'
end_time = datetime.today().strftime('%Y-%m-%d')
frequency = '1d'

logging.info("Fetching BTC PriceUSD...")
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
df_raw = df_raw.rename(columns={metric: 'PriceUSD'})
df_raw['time'] = pd.to_datetime(df_raw['time']).dt.normalize()
df_raw['time'] = df_raw['time'].dt.tz_localize(None)
df_raw.set_index('time', inplace=True)
df_raw = df_raw[['PriceUSD']]

# Sort index just in case (important for nearest index lookups)
df_raw.sort_index(inplace=True)


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
# Identify Highs and Lows
# ------------------------------------------------------------
all_time_highs = df_raw[df_raw['PriceUSD'] == df_raw['PriceUSD'].cummax()]

cycle_highs = df_raw.loc[df_raw.index.isin([
    df_raw.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmax()
    for i in range(1, len(halving_dates))
    if not df_raw.loc[halving_dates[i-1]:halving_dates[i]].empty
])]

cycle_lows = df_raw.loc[df_raw.index.isin([
    df_raw.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmin()
    for i in range(1, len(halving_dates))
    if not df_raw.loc[halving_dates[i-1]:halving_dates[i]].empty
])]


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_price_4_years_prior(current_date, df):
    """
    Returns the price from df nearest to (current_date - 4 years).
    If that date is before df's first date, returns None.
    """
    date_4y_prior = current_date - pd.DateOffset(years=4)
    if date_4y_prior < df.index[0]:
        return None
    
    # Use get_indexer with method='nearest'
    pos = df.index.get_indexer([date_4y_prior], method='nearest')[0]
    return df['PriceUSD'].iloc[pos]


def _days_since_halving(date, halving_dates):
    past_halvings = [halving for halving in halving_dates if halving <= date]
    if len(past_halvings) > 0:
        return (date - max(past_halvings)).days
    else:
        return np.nan


def _calculate_theta(date, halving_dates, fixed_halving):
    """
    1458 days ~ one full circle (2*pi).
    This approximates the ~4-year halving cycle in block-time.
    """
    days_since_fixed_halving = (date - fixed_halving).days
    return (-2 * np.pi * days_since_fixed_halving / 1458) + (np.pi / 2)


def _price_formatter(x, pos):
    real_value = 10**x
    if real_value >= 1e6:
        return f'${real_value * 1e-6:.0f}M'
    elif real_value >= 1e3:
        return f'${real_value * 1e-3:.0f}k'
    else:
        return f'${real_value:.0f}'


def _set_theta_labels(ax):
    # Example for labeling every 4 years around the circle
    years_90 = [2012 + 4 * i for i in range(5)]
    years_180 = [2011 + 4 * i for i in range(5)]
    years_270 = [2010 + 4 * i for i in range(5)]
    years_360 = [2009 + 4 * i for i in range(5)]
    
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=[years_360, years_90, years_180, years_270],
        fontsize=16,
        color='white'
    )
    
    theta_tick_labels = ax.get_xticklabels()
    # Adjust positions if desired
    theta_tick_labels[0].set_y(-0.4)
    theta_tick_labels[0].set_x(0.007)
    theta_tick_labels[2].set_y(-0.4)
    theta_tick_labels[2].set_x(-0.007)
    theta_tick_labels[1].set_y(-0.01)
    theta_tick_labels[3].set_y(-0.01)


def _create_polar_plot(df, halving_dates, fixed_halving):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='polar')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    r = np.log10(df['PriceUSD'])
    ax.set_title('Bitcoin Price Spiral', va='bottom', fontsize=32, color='white')
    
    ax.grid(True, color='gray', linestyle='--')
    ax.set_rticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_ylim(0, 6.1)  # Force the radial axis up to slightly above 10^6 so the $1M circle appears
    ax.yaxis.set_major_formatter(FuncFormatter(_price_formatter))
    ax.tick_params(axis='y', labelsize=16, colors='white')
    
    _set_theta_labels(ax)
    return fig, ax


# ------------------------------------------------------------
# Main Animation Function
# ------------------------------------------------------------
def animate_spiral_chart(df, duration=30, fps=60, pause_duration=2):
    """
    Renders a spiral chart of Bitcoin's price with halving events, cycle highs/lows,
    and 'price 4 years ago' data. Saves the result as a GIF in a 'gifs' folder.
    """
    total_frames = duration * fps
    pause_frames = pause_duration * fps

    # Identify marker indices from the relevant points
    ath_indices = df.index.get_indexer(all_time_highs.index, method='nearest')
    cycle_high_indices = df.index.get_indexer(cycle_highs.index, method='nearest')
    cycle_low_indices = df.index.get_indexer(cycle_lows.index, method='nearest')
    halving_indices = df.index.get_indexer(halving_dates, method='nearest')

    marker_indices = np.unique(
        np.concatenate([ath_indices, cycle_high_indices, cycle_low_indices, halving_indices])
    )
    
    # Build a unique set of indices to animate
    linear_indices = np.linspace(0, len(df) - 1, total_frames, dtype=int)
    indices = np.unique(np.concatenate([linear_indices, marker_indices]))

    # Create a copy after slicing to avoid SettingWithCopyWarning
    df = df.iloc[indices].copy()

    # Add extra columns
    df['Days_Since_Halving'] = df.index.to_series().apply(
        lambda x: _days_since_halving(x, halving_dates)
    )
    
    fixed_halving = pd.Timestamp('2024-04-20')
    df['Theta'] = df.index.to_series().apply(
        lambda x: _calculate_theta(x, halving_dates, fixed_halving)
    )

    # Create the polar plot
    fig, ax = _create_polar_plot(df, halving_dates, fixed_halving)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    line, = ax.plot([], [], lw=0.75, color='lightblue')

    # Halving markers
    halving_markers = []
    halving_labels = [
        '1st Halving (2012-11-28)',
        '2nd Halving (2016-07-09)',
        '3rd Halving (2020-05-11)',
        '4th Halving (2024-04-20)'
    ]
    for i, halving_date in enumerate(halving_dates):
        marker, = ax.plot([], [], 'ws', markersize=10, label=halving_labels[i])
        halving_markers.append(marker)

    # High/low markers
    all_time_high_markers, = ax.plot([], [], 'bo', markersize=10, markerfacecolor='none', label='All-Time High')
    cycle_high_markers, = ax.plot([], [], '^', markersize=10, color='orange', label='Cycle High')
    cycle_low_markers, = ax.plot([], [], 'ro', markersize=10, label='Cycle Low')

    fig.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1),
        fontsize=22,
        markerscale=2,
        frameon=False,
        facecolor='black',
        labelcolor='white'
    )

    # Radius: log10 of the price
    r = np.log10(df['PriceUSD'])

    # Position the text box further left
    price_text = ax.text(
        0.60, 0.02, '',
        transform=ax.transAxes,
        fontsize=28,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.7, pad=10)
    )

    def format_price(value):
        """
        Convert numeric price to human-readable string:
          e.g. 100000 -> "$100k"
        """
        if value is None or np.isnan(value):
            return 'N/A'
        elif value >= 1e6:
            return f'${value / 1e6:.0f}M'
        elif value >= 1e3:
            return f'${value / 1e3:.0f}k'
        else:
            return f'${value:.0f}'

    def init():
        line.set_data([], [])
        for marker in halving_markers:
            marker.set_data([], [])
        all_time_high_markers.set_data([], [])
        cycle_high_markers.set_data([], [])
        cycle_low_markers.set_data([], [])
        price_text.set_text('')
        return [
            line,
            *halving_markers,
            all_time_high_markers,
            cycle_high_markers,
            cycle_low_markers,
            price_text
        ]

    def update(frame):
        max_index = min(frame, len(df) - 1)

        # Plot line up to current index
        line.set_data(df['Theta'][:max_index], r[:max_index])

        current_date = df.index[max_index]
        current_date_str = current_date.strftime('%Y-%m-%d')
        current_price = df['PriceUSD'].iloc[max_index]

        # Retrieve price from 4 years ago via the *raw* DataFrame
        four_year_price = get_price_4_years_prior(current_date, df_raw)

        # Format both prices
        formatted_current = format_price(current_price)
        formatted_four_year = format_price(four_year_price)

        # Escape the $ to avoid LaTeX math mode issues:
        safe_current = formatted_current.replace('$', r'\$')
        safe_four_year = formatted_four_year.replace('$', r'\$')

        # Display logic:
        if four_year_price is None or formatted_four_year == 'N/A':
            # No valid price from 4 years ago
            price_text.set_text(
                f"Date: {current_date_str}\n"
                f"price {safe_current}"
            )
        else:
            # Display current plus 4-year price
            price_text.set_text(
                f"Date: {current_date_str}\n"
                f"price {safe_current} > price 4 yrs ago {safe_four_year}"
            )

        # ----------------------------------------------
        # Update halving markers
        # ----------------------------------------------
        for i, h_date in enumerate(halving_dates):
            if current_date >= h_date and h_date in df.index:
                halving_r = np.log10(df.loc[h_date, 'PriceUSD'])
                halving_theta = df.loc[h_date, 'Theta']
                halving_markers[i].set_data([halving_theta], [halving_r])
            else:
                halving_markers[i].set_data([], [])

        # ----------------------------------------------
        # Update All-Time High markers
        # ----------------------------------------------
        ath_dates_filtered = df.index[
            df.index.isin(all_time_highs.index) & (df.index <= current_date)
        ]
        if not ath_dates_filtered.empty:
            ath_r = np.log10(all_time_highs.loc[ath_dates_filtered, 'PriceUSD'])
            ath_theta = df.loc[ath_dates_filtered, 'Theta']
            all_time_high_markers.set_data(ath_theta, ath_r)

        # ----------------------------------------------
        # Update Cycle High markers
        # ----------------------------------------------
        cycle_high_dates_filtered = df.index[
            df.index.isin(cycle_highs.index) & (df.index <= current_date)
        ]
        if not cycle_high_dates_filtered.empty:
            cyc_high_r = np.log10(cycle_highs.loc[cycle_high_dates_filtered, 'PriceUSD'])
            cyc_high_theta = df.loc[cycle_high_dates_filtered, 'Theta']
            cycle_high_markers.set_data(cyc_high_theta, cyc_high_r)

        # ----------------------------------------------
        # Update Cycle Low markers
        # ----------------------------------------------
        cycle_low_dates_filtered = df.index[
            df.index.isin(cycle_lows.index) & (df.index <= current_date)
        ]
        if not cycle_low_dates_filtered.empty:
            cyc_low_r = np.log10(cycle_lows.loc[cycle_low_dates_filtered, 'PriceUSD'])
            cyc_low_theta = df.loc[cycle_low_dates_filtered, 'Theta']
            cycle_low_markers.set_data(cyc_low_theta, cyc_low_r)

        return [
            line,
            *halving_markers,
            all_time_high_markers,
            cycle_high_markers,
            cycle_low_markers,
            price_text
        ]

    # --------------------------------------------------------
    # Save Animation as GIF
    # --------------------------------------------------------
    gif_folder = "gifs"
    if not os.path.exists(gif_folder):
        os.makedirs(gif_folder)

    ani = FuncAnimation(
        fig, update,
        frames=len(df) + pause_frames,
        init_func=init,
        blit=True,
        interval=1000/fps,
        repeat=True
    )

    current_date_str = datetime.now().strftime("%b_%d_%H_%M").lower()
    filename = os.path.join(gif_folder, f"spiral_chart_{current_date_str}.gif")

    ani.save(filename, writer=PillowWriter(fps=fps), dpi=72)
    logging.info(f"Animation saved to: {filename}")

    plt.show()


# ------------------------------------------------------------
# Run the animation
# ------------------------------------------------------------
if __name__ == "__main__":
    # You can adjust duration, fps, pause_duration as needed
    animate_spiral_chart(df_raw, duration=10, fps=15, pause_duration=2)