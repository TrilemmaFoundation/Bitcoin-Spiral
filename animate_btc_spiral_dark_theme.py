import pandas as pd
import numpy as np
import os
from datetime import datetime
from coinmetrics.api_client import CoinMetricsClient
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation, PillowWriter

import time
from requests.exceptions import HTTPError

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Coin Metrics API client
client = CoinMetricsClient()

# Define the asset, metric, and time range
asset = 'btc'
metric = 'PriceUSD'
start_time = '2010-01-01'
end_time = datetime.today().strftime('%Y-%m-%d')
frequency = '1d'

logging.info("Fetching BTC PriceUSD...")
df = client.get_asset_metrics(
    assets=asset,
    metrics=[metric],
    frequency=frequency,
    start_time=start_time,
    end_time=end_time
).to_dataframe()

df = df.rename(columns={metric: 'PriceUSD'})
df['time'] = pd.to_datetime(df['time']).dt.normalize()
df['time'] = df['time'].dt.tz_localize(None)
df.set_index('time', inplace=True)
df = df[['PriceUSD']]

# Define halving dates
halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20')
]

all_time_highs = df[df['PriceUSD'] == df['PriceUSD'].cummax()]
cycle_highs = df.loc[df.index.isin([df.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmax() 
                                    for i in range(1, len(halving_dates)) if not df.loc[halving_dates[i-1]:halving_dates[i]].empty])]
cycle_lows = df.loc[df.index.isin([df.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmin() 
                                   for i in range(1, len(halving_dates)) if not df.loc[halving_dates[i-1]:halving_dates[i]].empty])]

def animate_spiral_chart(df, duration=30, fps=60, pause_duration=2):
    total_frames = duration * fps
    pause_frames = pause_duration * fps

    ath_indices = df.index.get_indexer(all_time_highs.index, method='nearest')
    cycle_high_indices = df.index.get_indexer(cycle_highs.index, method='nearest')
    cycle_low_indices = df.index.get_indexer(cycle_lows.index, method='nearest')
    halving_indices = df.index.get_indexer(halving_dates, method='nearest')

    marker_indices = np.unique(np.concatenate([ath_indices, cycle_high_indices, cycle_low_indices, halving_indices]))
    indices = np.unique(np.concatenate([np.linspace(0, len(df)-1, total_frames, dtype=int), marker_indices]))
    df = df.iloc[indices]
    df['Days_Since_Halving'] = df.index.to_series().apply(lambda x: _days_since_halving(x, halving_dates))

    fixed_halving = pd.Timestamp('2024-04-20')
    df['Theta'] = df.index.to_series().apply(lambda x: _calculate_theta(x, halving_dates, fixed_halving))

    fig, ax = _create_polar_plot(df, halving_dates, fixed_halving)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    line, = ax.plot([], [], lw=0.75, color='lightblue')

    halving_markers = []
    halving_labels = ['1st Halving (2012-11-28)', '2nd Halving (2016-07-09)', '3rd Halving (2020-05-11)', '4th Halving (2024-04-20)']

    for i, halving_date in enumerate(halving_dates):
        halving_marker, = ax.plot([], [], 'ws', markersize=10, label=halving_labels[i])
        halving_markers.append(halving_marker)

    all_time_high_markers, = ax.plot([], [], 'bo', markersize=10, markerfacecolor='none', label='All-Time High')
    cycle_high_markers, = ax.plot([], [], '^', markersize=10, color='orange', label='Cycle High')
    cycle_low_markers, = ax.plot([], [], 'ro', markersize=10, label='Cycle Low')

    fig.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=22, markerscale=2, frameon=False, facecolor='black', labelcolor='white')

    r = np.log10(df['PriceUSD'])
    price_text = ax.text(1, 0.05, '', transform=ax.transAxes, fontsize=28, color='white',  
                         verticalalignment='bottom', horizontalalignment='left',  
                         bbox=dict(facecolor='black', alpha=0.7, pad=10))  

    def format_price(value):
        if value >= 1e6:
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
        return [line] + halving_markers + [all_time_high_markers, cycle_high_markers, cycle_low_markers, price_text]

    def update(frame):
        max_index = min(frame, len(df) - 1)
        
        line.set_data(df['Theta'][:max_index], r[:max_index])
        current_date = df.index[max_index].strftime('%Y-%m-%d')
        current_price = df['PriceUSD'].iloc[max_index]
        formatted_price = format_price(current_price)
        price_text.set_text(f'Date: {current_date}\nPrice: {formatted_price}')

        for i, halving_date in enumerate(halving_dates):
            if df.index[max_index] >= halving_date:
                halving_r = np.log10(df.loc[halving_date, 'PriceUSD'])
                halving_theta = df.loc[halving_date, 'Theta']
                halving_markers[i].set_data([halving_theta], [halving_r])
            else:
                halving_markers[i].set_data([], [])

        ath_dates_filtered = df.index[df.index.isin(all_time_highs.index) & (df.index <= df.index[max_index])]
        if not ath_dates_filtered.empty:
            ath_r = np.log10(all_time_highs.loc[ath_dates_filtered, 'PriceUSD'])
            ath_theta = df.loc[ath_dates_filtered, 'Theta']
            all_time_high_markers.set_data(ath_theta, ath_r)

        cycle_high_dates_filtered = df.index[df.index.isin(cycle_highs.index) & (df.index <= df.index[max_index])]
        if not cycle_high_dates_filtered.empty:
            cycle_high_r = np.log10(cycle_highs.loc[cycle_high_dates_filtered, 'PriceUSD'])
            cycle_high_theta = df.loc[cycle_high_dates_filtered, 'Theta']
            cycle_high_markers.set_data(cycle_high_theta, cycle_high_r)

        cycle_low_dates_filtered = df.index[df.index.isin(cycle_lows.index) & (df.index <= df.index[max_index])]
        if not cycle_low_dates_filtered.empty:
            cycle_low_r = np.log10(cycle_lows.loc[cycle_low_dates_filtered, 'PriceUSD'])
            cycle_low_theta = df.loc[cycle_low_dates_filtered, 'Theta']
            cycle_low_markers.set_data(cycle_low_theta, cycle_low_r)

        return [line] + halving_markers + [all_time_high_markers, cycle_high_markers, cycle_low_markers, price_text]

    gif_folder = "gifs"
    if not os.path.exists(gif_folder):
        os.makedirs(gif_folder)

    ani = FuncAnimation(
        fig, update, frames=len(df) + pause_frames, init_func=init, blit=True, interval=1000/fps, repeat=True
    )

    current_date = datetime.now().strftime("%b_%d_%H_%M").lower()
    filename = os.path.join(gif_folder, f"spiral_chart_{current_date}.gif")
    ani.save(filename, writer=PillowWriter(fps=fps), dpi=72)

    plt.show()

def _days_since_halving(date, halving_dates):
    past_halvings = [halving for halving in halving_dates if halving <= date]
    if len(past_halvings) > 0:
        return (date - max(past_halvings)).days
    else:
        return np.nan

def _calculate_theta(date, halving_dates, fixed_halving):
    days_since_fixed_halving = (date - fixed_halving).days
    return (-2 * np.pi * days_since_fixed_halving / 1458) + (np.pi / 2)

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
    ax.yaxis.set_major_formatter(FuncFormatter(_price_formatter))
    ax.tick_params(axis='y', labelsize=16, colors='white')
    
    _set_theta_labels(ax)
    return fig, ax

def _price_formatter(x, pos):
    real_value = 10**x
    if real_value >= 1e6:
        return f'${real_value * 1e-6:.0f}M'
    elif real_value >= 1e3:
        return f'${real_value * 1e-3:.0f}k'
    else:
        return f'${real_value:.0f}'

def _set_theta_labels(ax):
    years_90 = [2012 + 4 * i for i in range(5)]
    years_180 = [2011 + 4 * i for i in range(5)]
    years_270 = [2010 + 4 * i for i in range(5)]
    years_360 = [2009 + 4 * i for i in range(5)]
    
    ax.set_thetagrids([0, 90, 180, 270], labels=[years_360, years_90, years_180, years_270], fontsize=16, color='white') 
    
    theta_tick_labels = ax.get_xticklabels()
    theta_tick_labels[0].set_y(-0.4)
    theta_tick_labels[0].set_x(0.007)
    theta_tick_labels[2].set_y(-0.4)
    theta_tick_labels[2].set_x(-0.007)
    theta_tick_labels[1].set_y(-0.01)
    theta_tick_labels[3].set_y(-0.01)

animate_spiral_chart(df, duration=10, fps=15, pause_duration=2)