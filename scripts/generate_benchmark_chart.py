#!/usr/bin/env python3
"""Generate benchmark visualization for social media."""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with dark theme for better contrast
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

# Colors
e3nn_color = '#6e7681'
batmobile_color = '#58a6ff'
speedup_color = '#3fb950'

# Data: Operation benchmarks
operations = ['Spherical\nHarmonics', 'Tensor\nProduct', 'TP\nBackward', 'Fused\nSH+TP']
e3nn_times = [0.142, 1.847, 3.21, 0.574]  # ms
batmobile_times = [0.012, 0.089, 0.156, 0.413]  # ms
speedups = [11.8, 20.8, 20.6, 1.39]

x = np.arange(len(operations))
width = 0.35

# Left plot: Time comparison (log scale)
ax1.set_facecolor('#0d1117')
bars1 = ax1.bar(x - width/2, e3nn_times, width, label='e3nn', color=e3nn_color, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + width/2, batmobile_times, width, label='batmobile', color=batmobile_color, edgecolor='white', linewidth=0.5)

ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Kernel Execution Time', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(operations, fontsize=10)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_yscale('log')
ax1.set_ylim(0.005, 5)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add time labels on bars
for bar, time in zip(bars1, e3nn_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{time:.2f}', ha='center', va='bottom', fontsize=8, color='white')
for bar, time in zip(bars2, batmobile_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{time:.3f}', ha='center', va='bottom', fontsize=8, color='white')

# Right plot: Speedup bars
ax2.set_facecolor('#0d1117')
bars3 = ax2.bar(x, speedups, width=0.6, color=speedup_color, edgecolor='white', linewidth=0.5)

ax2.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup vs e3nn', fontsize=14, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(operations, fontsize=10)
ax2.axhline(y=1, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax2.set_ylim(0, 25)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add speedup labels on bars
for bar, speedup in zip(bars3, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{speedup:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

# Main title
fig.suptitle('batmobile: CUDA Kernels for Equivariant GNNs',
             fontsize=16, fontweight='bold', y=0.98, color='white')

# Subtitle
fig.text(0.5, 0.02, 'RTX 3090 | N=1000 atoms | C=32 channels | L_max=3',
         ha='center', fontsize=10, color='#8b949e')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save to project directory and blog directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
blog_dir = '/home/infatoshi/elliotarledge.com/blogs/Batmobile'

for output_dir in [project_dir, blog_dir]:
    plt.savefig(os.path.join(output_dir, 'benchmark_chart.png'), dpi=150, facecolor='#0d1117', edgecolor='none',
                bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output_dir}/benchmark_chart.png (dark theme)")

# Also create a simpler "hero" image
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Horizontal bar chart - more impactful
y_pos = np.arange(len(operations))
bars = ax.barh(y_pos, speedups, color=speedup_color, edgecolor='white', linewidth=0.5, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(operations, fontsize=12)
ax.set_xlabel('Speedup vs e3nn', fontsize=12, fontweight='bold')
ax.set_xlim(0, 25)
ax.axvline(x=1, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add speedup labels
for bar, speedup in zip(bars, speedups):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{speedup:.1f}x', ha='left', va='center', fontsize=14, fontweight='bold', color='white')

ax.set_title('batmobile: 10-20x Faster Equivariant GNN Kernels',
             fontsize=16, fontweight='bold', pad=15, color='white')

plt.tight_layout()

for output_dir in [project_dir, blog_dir]:
    plt.savefig(os.path.join(output_dir, 'benchmark_hero.png'), dpi=150, facecolor='#0d1117', edgecolor='none',
                bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output_dir}/benchmark_hero.png (hero image)")

print("\nDone! Use benchmark_hero.png for the main social media image.")
