#!/usr/bin/env python3
import os
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from io import StringIO

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm

# --- Data Extraction & Loading ---

def extract_btc_data_to_csv(local_path='btc_data.csv'):
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    resp = requests.get(url)
    resp.raise_for_status()
    btc_df = pd.read_csv(StringIO(resp.text))
    btc_df['time'] = (
        pd.to_datetime(btc_df['time'])
          .dt.normalize()
          .dt.tz_localize(None)
    )
    btc_df.set_index('time', inplace=True)
    btc_df.to_csv(local_path)

def load_data(price_path="btc_data.csv", weights_path="200ma_strategy_weights.csv"):
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    price_df = price_df.sort_index().loc[~price_df.index.duplicated(keep='last'), ['PriceUSD']]

    w_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    w_df = w_df.sort_index().loc[~w_df.index.duplicated(keep='last')]
    if w_df.columns[0] != 'Weight':
        w_df.rename(columns={w_df.columns[0]: 'Weight'}, inplace=True)

    return price_df.join(w_df, how='inner')

# --- Helpers ---

halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def _days_since_halving(date, halvings):
    past = [h for h in halvings if h <= date]
    return (date - max(past)).days if past else np.nan

def _calculate_theta(date, halvings, fixed):
    dsf = (date - fixed).days
    return -2 * np.pi * dsf / 1458 + np.pi/2

def _price_formatter(x, pos):
    v = 10**x
    if v >= 1e6: return f'${v*1e-6:.0f}M'
    if v >= 1e3: return f'${v*1e-3:.0f}k'
    return f'${v:.0f}'

def _set_theta_labels(ax):
    yrs90  = [2012 + 4*i for i in range(5)]
    yrs180 = [2011 + 4*i for i in range(5)]
    yrs270 = [2010 + 4*i for i in range(5)]
    yrs360 = [2009 + 4*i for i in range(5)]
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=[yrs360, yrs90, yrs180, yrs270],
        fontsize=16, color='white'
    )
    tl = ax.get_xticklabels()
    tl[0].set_y(-0.4); tl[0].set_x(0.007)
    tl[2].set_y(-0.4); tl[2].set_x(-0.007)
    tl[1].set_y(-0.01); tl[3].set_y(-0.01)

# --- Plot Setup ---

def _create_polar_plot():
    fig = plt.figure(figsize=(16,12))
    ax  = fig.add_subplot(111, projection='polar')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)
    ax.set_position([0.05, 0.10, 0.90, 0.80])
    ax.set_title("Bitcoin Accumulation Spiral", va='bottom',
                 fontsize=28, color='white')
    ax.grid(True, color='gray', linestyle='--')
    ax.set_rticks([0,1,2,3,4,5,6])
    ax.yaxis.set_major_formatter(FuncFormatter(_price_formatter))
    ax.tick_params(axis='y', labelsize=16, colors='white')
    _set_theta_labels(ax)
    return fig, ax

# --- Animation ---

def animate_spiral_chart(df):
    # compute extremes
    ath = df[df['PriceUSD'] == df['PriceUSD'].cummax()]
    ch  = df.loc[df.index.isin([
        df.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmax()
        for i in range(1, len(halving_dates))
        if not df.loc[halving_dates[i-1]:halving_dates[i]].empty
    ])]
    cl  = df.loc[df.index.isin([
        df.loc[halving_dates[i-1]:halving_dates[i]]['PriceUSD'].idxmin()
        for i in range(1, len(halving_dates))
        if not df.loc[halving_dates[i-1]:halving_dates[i]].empty
    ])]

    idx = df.index
    markers = np.unique(np.concatenate([
        idx.get_indexer(ath.index, 'nearest'),
        idx.get_indexer(ch.index, 'nearest'),
        idx.get_indexer(cl.index, 'nearest'),
        idx.get_indexer(halving_dates, 'nearest'),
    ]))

    seq = np.unique(np.concatenate([np.arange(len(df)), markers]))
    df2 = df.iloc[seq].copy()
    fixed_h = halving_dates[-1]
    df2['Days_Since_Halving'] = df2.index.to_series().map(
        lambda d: _days_since_halving(d, halving_dates))
    df2['Theta'] = df2.index.to_series().map(
        lambda d: _calculate_theta(d, halving_dates, fixed_h))
    r = np.log10(df2['PriceUSD'])

    total_frames = len(df2)
    fps = total_frames / 30.0  # ~30-second output

    fig, ax = _create_polar_plot()
    w = df2['Weight'].replace(0, np.nan)
    norm = LogNorm(vmin=w.min(), vmax=w.max())
    scatter = ax.scatter([], [], s=8, c=[], cmap='plasma', norm=norm)

    cbar = fig.colorbar(scatter, ax=ax,
                        orientation='horizontal',
                        fraction=0.04, pad=0.05, shrink=0.8)
    cbar.ax.set_position([0.10, 0.06, 0.80, 0.02])
    cbar.set_label('Budget Allocation Weight (log scale)',
                   color='white', fontsize=16)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.tick_bottom()
    cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')
    cbar.outline.set_edgecolor('white')

    halver_markers = []
    for lbl in [
        '1st Halving (2012-11-28)',
        '2nd Halving (2016-07-09)',
        '3rd Halving (2020-05-11)',
        '4th Halving (2024-04-20)'
    ]:
        m, = ax.plot([], [], 'ws', markersize=10, label=lbl)
        halver_markers.append(m)
    ath_m, = ax.plot([], [], 'bo', markersize=10,
                     markerfacecolor='none', label='All-Time High')
    ch_m,  = ax.plot([], [], '^', markersize=10,
                     color='orange', label='Cycle High')
    cl_m,  = ax.plot([], [], 'ro', markersize=10, label='Cycle Low')

    leg = fig.legend(loc='upper left', bbox_to_anchor=(0,1),
                     fontsize=22, markerscale=2, frameon=False)
    for txt in leg.get_texts(): txt.set_color('white')
    leg.get_frame().set_facecolor('black')

    price_text = ax.text(1, 0.05, '', transform=ax.transAxes,
                         fontsize=28, color='white',
                         va='bottom', ha='left',
                         bbox=dict(facecolor='black', alpha=0.7, pad=10))

    def init():
        scatter.set_offsets(np.empty((0,2)))
        scatter.set_array(np.array([]))
        for m in halver_markers: m.set_data([], [])
        ath_m.set_data([], []); ch_m.set_data([], []); cl_m.set_data([], [])
        price_text.set_text('')
        return [scatter] + halver_markers + [ath_m, ch_m, cl_m, price_text]

    def update(frame):
        coords = np.column_stack((df2['Theta'][:frame+1], r[:frame+1]))
        scatter.set_offsets(coords)
        scatter.set_array(df2['Weight'].values[:frame+1])

        date = df2.index[frame].strftime('%Y-%m-%d')
        price = df2['PriceUSD'].iloc[frame]
        price_text.set_text(f'Date: {date}\nPrice: ${price:,.0f}')

        for i, hd in enumerate(halving_dates):
            if df2.index[frame] >= hd:
                t_h = df2.loc[hd, 'Theta']
                r_h = np.log10(df2.loc[hd, 'PriceUSD'])
                halver_markers[i].set_data([t_h], [r_h])
            else:
                halver_markers[i].set_data([], [])

        for src, marker in [(ath, ath_m), (ch, ch_m), (cl, cl_m)]:
            dates = df2.index[df2.index.isin(src.index) & 
                              (df2.index <= df2.index[frame])]
            if not dates.empty:
                t  = df2.loc[dates, 'Theta']
                rv = np.log10(src.loc[dates, 'PriceUSD'])
                marker.set_data(t, rv)

        return [scatter] + halver_markers + [ath_m, ch_m, cl_m, price_text]

    os.makedirs("videos", exist_ok=True)
    out = os.path.join("videos", f"spiral_{datetime.now():%b_%d_%H_%M}.mp4")
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    ani = FuncAnimation(fig, update, frames=total_frames,
                        init_func=init, blit=True,
                        interval=1000/fps)
    ani.save(out, writer=writer, dpi=144)
    plt.show()

if __name__ == "__main__":
    extract_btc_data_to_csv("btc_data.csv")
    df = load_data("btc_data.csv", "200ma_strategy_weights.csv")
    animate_spiral_chart(df)