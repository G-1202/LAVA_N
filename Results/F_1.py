import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

colors = [(131/255, 182/255, 225/255),
          (245/255, 220/255, 117/255),
          (160/255, 210/255, 147/255),
          (160/255, 173/255, 208/255),
          (235/255, 167/255, 100/255),
          (240/255, 118/255, 115/255)]

acc_lat = [[0.7502, 0.024766, 4.8424, 0.48303],
           [0.76472, 0.025861, 6.3787, 0.69142],
           [0.75297, 0.029189, 4.5878, 1.1557],
           [0.7324, 0.021695, 2.9508, 0.52083],
           [0.75393, 0.024973, 2.5239, 0.71649],
           [0.83056, 0.015039, 1.6692, 0.17673]]

qoa_means = [0.26833, 0.12406, 0.30936, 0.44844, 0.52336, 0.66796]
qoa_stds = [0.47839, 0.49531, 0.48024, 0.45333, 0.425510, 0.38864]

def F_1():
    labels = ['AWStream', 'DDS', 'DAO', 'CASVA', 'FHVAC', 'LAVA']
    markers = ['^', 'o', 's', 'D','H', 'v']
    markers_size = [25, 24, 24, 24, 28, 25, 25]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    legend_handles = []
    # ----------------- No.1 -----------------
    for ind, item in enumerate(labels):
        accuracy_mean = acc_lat[ind][0]
        accuracy_std = acc_lat[ind][1]
        latency_mean = acc_lat[ind][2]
        latency_std = acc_lat[ind][3]
        errorbar_plot = ax1.errorbar(
            latency_mean, accuracy_mean,
            xerr=latency_std,
            yerr=accuracy_std,
            fmt=markers[ind],
            color=colors[ind],
            ecolor=colors[ind],
            markersize=markers_size[ind],
            markeredgecolor='black',
            label=labels[ind],
            capsize = 7,
            elinewidth = 4,
            capthick = 4
        )
        legend_handles.append(errorbar_plot[0])
    ax1.set_xlabel('Latency (s)', fontsize=22)
    ax1.set_ylabel('Accuracy', fontsize=22)
    ax1.tick_params(axis='both', labelsize=22)
    ax1.set_ylim(0.69, 0.86)
    ax1.set_yticks([0.70, 0.75, 0.8, 0.85])
    ax1.set_xlim(1, 7.5)
    ax1.set_xticks([1, 3, 5, 7])
    ax1.grid(True)

    # Add an arrow
    ax1.annotate('Better',
                 xy=(2.2, 0.83),
                 xytext=(3.4, 0.79),
                 arrowprops=dict(facecolor=(28 / 255, 60 / 255, 99 / 255),
                                 shrink=0.05,
                                 width=6,
                                 headwidth=3 * 6,
                                 headlength=4 * 6),
                 fontsize=22)
    # ----------------- No.2 -----------------
    for ind in range(len(labels)):
        ax2.errorbar(
            x=ind,
            y=qoa_means[ind],
            yerr=qoa_stds[ind],
            fmt=markers[ind],
            color=colors[ind],
            ecolor=colors[ind],
            markersize=markers_size[ind],
            markeredgecolor='black',
            capsize=7,
            elinewidth=4,
            capthick=4
        )
    ax2.set_ylim(-0.5, 1.1)
    ax2.set_yticks([-0.5, -0.2, 0.1, 0.4, 0.7, 1])
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticklabels(labels, fontsize=22, ha='center', rotation=30)
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylabel('QoA', fontsize=22)
    ax2.tick_params(axis='both', labelsize=22)
    ax2.grid(True, zorder=1)

    fig.legend(
        handles=legend_handles,
        labels=labels,
        loc='upper center',
        ncol=6,
        fontsize=22,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.5,
        frameon=False
    )

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'F_1.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':
    F_1()