import pandas as pd
import numpy as np
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
metric = 'ReferenceRate'
start_time = '2010-01-01'
end_time = datetime.today().strftime('%Y-%m-%d')  # Set end_time to today's date
frequency = '1d'

# Fetch the metric data for the specified asset and time range
logging.info("Fetching BTC ReferenceRate...")
df = client.get_asset_metrics(
    assets=asset,
    metrics=[metric],
    frequency=frequency,
    start_time=start_time,
    end_time=end_time
).to_dataframe()

# Rename the 'ReferenceRate' column to 'Close'
df = df.rename(columns={metric: 'Close'})

# Set 'time' as the index and normalize to remove the time component but keep it as a DatetimeIndex
df['time'] = pd.to_datetime(df['time']).dt.normalize()

# Remove timezone information, if any
df['time'] = df['time'].dt.tz_localize(None)

# Set 'time' as the index
df.set_index('time', inplace=True)

# Only keep close data
df = df[['Close']]

# Define halving dates
halving_dates = [
    pd.Timestamp('2012-11-28'),  # 1st halving
    pd.Timestamp('2016-07-09'),  # 2nd halving
    pd.Timestamp('2020-05-11'),  # 3rd halving
    pd.Timestamp('2024-04-20')   # 4th halving
]

# Find the all-time highs, cycle highs, and cycle lows
all_time_highs = df[df['Close'] == df['Close'].cummax()]
cycle_highs = df.loc[df.index.isin([df.loc[halving_dates[i-1]:halving_dates[i]]['Close'].idxmax() for i in range(1, len(halving_dates))])]
cycle_lows = df.loc[df.index.isin([df.loc[halving_dates[i-1]:halving_dates[i]]['Close'].idxmin() for i in range(1, len(halving_dates))])]

def animate_spiral_chart(df, duration=30, fps=60, pause_duration=2):
    """
    Creates an animated spiral chart of Bitcoin prices with respect to halving events.
    The animation shows the series gradually being plotted, with a pause at the last frame before restarting.
    Includes a legend for halving events, all-time highs, cycle highs, and cycle lows.
    Displays live price and date dynamically during the animation.
    
    Args:
    df (pd.DataFrame): DataFrame with datetime index and 'Close' column representing daily Bitcoin prices.
    duration (int): Total duration of the animation in seconds (excluding the pause).
    fps (int): Frames per second for the animation.
    pause_duration (int): Duration to pause on the last frame in seconds.
    """
    total_frames = duration * fps
    pause_frames = pause_duration * fps  # Number of frames for the pause

    # Get the indices for all markers
    ath_indices = df.index.get_indexer(all_time_highs.index, method='nearest')
    cycle_high_indices = df.index.get_indexer(cycle_highs.index, method='nearest')
    cycle_low_indices = df.index.get_indexer(cycle_lows.index, method='nearest')
    halving_indices = df.index.get_indexer(halving_dates, method='nearest')

    # Combine all marker indices to ensure they're included
    marker_indices = np.unique(np.concatenate([ath_indices, cycle_high_indices, cycle_low_indices, halving_indices]))

    # Reduce number of frames but ensure all marker indices are included
    indices = np.unique(np.concatenate([np.linspace(0, len(df)-1, total_frames, dtype=int), marker_indices]))
    df = df.iloc[indices]

    # Calculate days since the last halving
    df['Days_Since_Halving'] = df.index.to_series().apply(lambda x: _days_since_halving(x, halving_dates))

    # Calculate theta with the 2024 halving fixed at pi/2
    fixed_halving = pd.Timestamp('2024-04-20')
    df['Theta'] = df.index.to_series().apply(lambda x: _calculate_theta(x, halving_dates, fixed_halving))

    # Create the spiral plot
    fig, ax = _create_polar_plot(df, halving_dates, fixed_halving)

    # Initialize empty plot for animation
    line, = ax.plot([], [], lw=0.75, color='blue')

    # Initialize individual halving markers and labels
    halving_markers = []
    halving_labels = ['1st Halving (2012-11-28)', '2nd Halving (2016-07-09)', '3rd Halving (2020-05-11)', '4th Halving (2024-04-20)']

    for i, halving_date in enumerate(halving_dates):
        # Create a marker for each halving
        halving_marker, = ax.plot([], [], 'ks', markersize=10, label=halving_labels[i])
        halving_markers.append(halving_marker)

    # Initialize markers for other key points
    all_time_high_markers, = ax.plot([], [], 'bo', markersize=10, markerfacecolor='none', label='All-Time High')
    cycle_high_markers, = ax.plot([], [], '^', markersize=10, color='orange', label='Cycle High')
    cycle_low_markers, = ax.plot([], [], 'ro', markersize=10, label='Cycle Low')

    # Add a legend with all the markers
    ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 1))

    # Convert price to log10 for scaling
    r = np.log10(df['Close'])

    # Add text to display live price and date
    price_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black', 
                         bbox=dict(facecolor='white', alpha=0.8))
    
    def format_price(value):
        """Format price into human-readable format."""
        if value >= 1e6:
            return f'${value / 1e6:.0f}M'
        elif value >= 1e3:
            return f'${value / 1e3:.0f}k'
        else:
            return f'${value:.0f}'

    def init():
        """Initialize the background of the plot."""
        line.set_data([], [])
        for marker in halving_markers:
            marker.set_data([], [])
        all_time_high_markers.set_data([], [])
        cycle_high_markers.set_data([], [])
        cycle_low_markers.set_data([], [])
        price_text.set_text('')  # Initialize with no text
        return [line] + halving_markers + [all_time_high_markers, cycle_high_markers, cycle_low_markers, price_text]

    def update(frame):
        """Update the plot with each new frame for the animation."""
        max_index = min(frame, len(df) - 1)  # Ensure the frame index doesn't go out of bounds
        
        # Update the price line incrementally
        line.set_data(df['Theta'][:max_index], r[:max_index])

        # Update live price and date text
        current_date = df.index[max_index].strftime('%Y-%m-%d')
        current_price = df['Close'].iloc[max_index]
        formatted_price = format_price(current_price)  # Format the price
        price_text.set_text(f'Date: {current_date}\nPrice: {formatted_price}')
        
        # Display halving markers incrementally for each halving event
        for i, halving_date in enumerate(halving_dates):
            if df.index[max_index] >= halving_date:  # Show the marker when time reaches the halving date
                halving_r = np.log10(df.loc[halving_date, 'Close'])
                halving_theta = df.loc[halving_date, 'Theta']
                halving_markers[i].set_data([halving_theta], [halving_r])
            else:
                halving_markers[i].set_data([], [])  # Hide the marker if not reached yet

        # Display all-time highs incrementally
        ath_dates_filtered = df.index[df.index.isin(all_time_highs.index) & (df.index <= df.index[max_index])]
        if not ath_dates_filtered.empty:
            ath_r = np.log10(all_time_highs.loc[ath_dates_filtered, 'Close'])
            ath_theta = df.loc[ath_dates_filtered, 'Theta']
            all_time_high_markers.set_data(ath_theta, ath_r)

        # Display cycle highs incrementally
        cycle_high_dates_filtered = df.index[df.index.isin(cycle_highs.index) & (df.index <= df.index[max_index])]
        if not cycle_high_dates_filtered.empty:
            cycle_high_r = np.log10(cycle_highs.loc[cycle_high_dates_filtered, 'Close'])
            cycle_high_theta = df.loc[cycle_high_dates_filtered, 'Theta']
            cycle_high_markers.set_data(cycle_high_theta, cycle_high_r)

        # Display cycle lows incrementally
        cycle_low_dates_filtered = df.index[df.index.isin(cycle_lows.index) & (df.index <= df.index[max_index])]
        if not cycle_low_dates_filtered.empty:
            cycle_low_r = np.log10(cycle_lows.loc[cycle_low_dates_filtered, 'Close'])
            cycle_low_theta = df.loc[cycle_low_dates_filtered, 'Theta']
            cycle_low_markers.set_data(cycle_low_theta, cycle_low_r)

        return [line] + halving_markers + [all_time_high_markers, cycle_high_markers, cycle_low_markers, price_text]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(df) + pause_frames, init_func=init, blit=True, interval=1000/fps, repeat=True
    )

    # Save the animation as a GIF
    current_date = datetime.now().strftime("%b_%d_%H_%M").lower()  # Includes hours and minutes
    filename = f"spiral_chart_{current_date}.gif"
    ani.save(filename, writer=PillowWriter(fps=fps))

    # Show the plot
    plt.show()

# Define auxiliary functions
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
    r = np.log10(df['Close'])
    ax.set_title('Bitcoin Price Spiral', va='bottom')
    ax.grid(True)
    ax.set_rticks([0, 1, 2, 3, 4, 5, 6])
    ax.yaxis.set_major_formatter(FuncFormatter(_price_formatter))
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
    ax.set_thetagrids([0, 90, 180, 270], labels=[years_360, years_90, years_180, years_270])
    theta_tick_labels = ax.get_xticklabels()
    theta_tick_labels[0].set_y(-0.3)
    theta_tick_labels[2].set_y(-0.3)
    theta_tick_labels[1].set_y(-0.01)
    theta_tick_labels[3].set_y(-0.01)

# Call the function to create and animate the spiral chart with dynamic markers and legend
animate_spiral_chart(df, duration=30, fps=60, pause_duration=2)