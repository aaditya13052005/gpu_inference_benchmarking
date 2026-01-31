import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# CONFIG
# ------------------------------
CSV_PATH = '../results/data/benchmark_results.csv'
OUTPUT_DIR = '../results/plots/'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(CSV_PATH)

# Ensure batch_size is sorted numerically
df['batch_size'] = df['batch_size'].astype(int)
df.sort_values(['model', 'batch_size', 'device', 'precision'], inplace=True)

# ------------------------------
# PLOT 1: Latency vs Batch Size
# ------------------------------
def plot_latency_vs_batch(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    devices = ['cpu', 'cuda']
    for i, device in enumerate(devices):
        ax = axes[i]
        for model in df['model'].unique():
            data = df[(df['device']==device) & (df['model']==model)]
            ax.plot(data['batch_size'], data['mean_latency_ms'], marker='o', label=model)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'{device.upper()} Inference Latency')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}latency_vs_batch.png', dpi=300)
    print(f'✅ Saved latency_vs_batch.png')

# ------------------------------
# PLOT 2: Throughput Scaling
# ------------------------------
def plot_throughput_scaling(df):
    fig, ax = plt.subplots(figsize=(10,6))
    gpu_data = df[df['device']=='cuda']
    for model in gpu_data['model'].unique():
        for precision in ['fp32','fp16']:
            data = gpu_data[(gpu_data['model']==model) & (gpu_data['precision']==precision)]
            ax.plot(data['batch_size'], data['throughput_img_per_sec'], marker='o', label=f'{model}-{precision}')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (images/sec)')
    ax.set_title('GPU Throughput Scaling')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}throughput_scaling.png', dpi=300)
    print(f'✅ Saved throughput_scaling.png')

# ------------------------------
# PLOT 3: FP16 Speedup Heatmap
# ------------------------------
def plot_fp16_speedup_heatmap(df):
    models = df['model'].unique()
    batch_sizes = sorted(df['batch_size'].unique())
    speedup_matrix = np.zeros((len(models), len(batch_sizes)))

    for i, model in enumerate(models):
        for j, bs in enumerate(batch_sizes):
            fp32_time = df[(df['model']==model) & (df['batch_size']==bs) & (df['precision']=='fp32')]['mean_latency_ms'].values[0]
            fp16_time = df[(df['model']==model) & (df['batch_size']==bs) & (df['precision']=='fp16')]['mean_latency_ms'].values[0]
            speedup_matrix[i,j] = fp32_time / fp16_time

    plt.figure(figsize=(10,6))
    sns.heatmap(speedup_matrix, annot=True, fmt='.2f',
                xticklabels=batch_sizes, yticklabels=models,
                cmap='RdYlGn', vmin=1.0,
                cbar_kws={'label':'Speedup (FP32/FP16)'})
    plt.title('FP16 Speedup Across Models and Batch Sizes')
    plt.xlabel('Batch Size')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fp16_speedup_heatmap.png', dpi=300)
    print(f'✅ Saved fp16_speedup_heatmap.png')

# ------------------------------
# MAIN
# ------------------------------
if __name__ == '__main__':
    plot_latency_vs_batch(df)
    plot_throughput_scaling(df)
    plot_fp16_speedup_heatmap(df)
    print('\n🎉 All plots generated! Check results/plots/')
