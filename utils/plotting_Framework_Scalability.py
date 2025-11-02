import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.font_manager import FontProperties

label_map = {
    'CrossQ': 'CrossQ',
    'FBiT(CrossQ)': 'FBiT(CrossQ)',
}

experiment_order = [
    'FBiT(CrossQ)',
    'CrossQ',
]

palette = "tab10"
x_axis_label = "Training Step"
y_axis_label = "Cumulative Return"
DOWNSAMPLE_STEP = 5

model_lower_bounds = {
    'model1': -5000,
    'model2': -10000,
    'model5': -20000
}
color_map = {
    'FBiT(CrossQ)': '#9467bd',  
    'CrossQ': '#e377c2'
}
root_dir = "results/SupplyChainGReal-v0/total_results"



def extract_model(filename):
    match = re.search(r"model(\d+)", filename)
    return f"model{match.group(1)}" if match else None


def downsample_data(steps, values, step_size=DOWNSAMPLE_STEP):
    downsampled_steps, downsampled_values = [], []
    for i in range(0, len(steps), step_size):
        group_steps = steps[i:i + step_size]
        group_values = values[i:i + step_size]
        downsampled_steps.append(np.mean(group_steps))
        downsampled_values.append(np.mean(group_values))
    return np.array(downsampled_steps), np.array(downsampled_values)


def format_k(x, pos):
    if x == 0:
        return "0"
    return f"{x / 1000:.1f}K" if (x % 1000) != 0 else f"{int(x / 1000)}K"


def generate_multi_model_plot():
    model_algo_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # 加载数据
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if not filename.endswith(".csv"):
            continue
        try:
            model = extract_model(filename)
            if not model:
                print(f"警告：{filename} 未匹配到Model（跳过）")
                continue
            algo = next((k for k in label_map if k in filename), None)
            if not algo:
                print(f"警告：{filename} 未匹配到算法（跳过）")
                continue
            df = pd.read_csv(file_path)
            if "Step" not in df.columns or "Value" not in df.columns:
                print(f"警告：{filename} 缺少Step或Value列（跳过）")
                continue
            for _, row in df.iterrows():
                step, value = row["Step"], row["Value"]
                if not np.isnan(step) and not np.isnan(value):
                    model_algo_data[model][algo][step].append(value)
        except Exception as e:
            print(f"处理 {filename} 失败: {e}")

    if not model_algo_data:
        print("错误：未找到有效数据！")
        return

    target_models = ['model1', 'model2', 'model5']
    missing_models = [m for m in target_models if m not in model_algo_data]
    if missing_models:
        print(f"错误：以下Model未在数据中找到：{missing_models}")
        return
    print(f"将绘制的Model：{target_models}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 12,
        "figure.dpi": 300,
        "lines.linewidth": 1.5,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    bold_font = FontProperties(weight='bold')

    legend_labels = []
    legend_handles = []

    for ax_idx, target_model in enumerate(target_models):
        ax = axes[ax_idx]
        algo_data = model_algo_data[target_model]
        if not algo_data:
            print(f"警告：{target_model} 无有效数据（跳过）")
            continue

        plot_data = {}
        max_value = -np.inf
        max_step = 0
        for algo_folder, step_values in algo_data.items():
            algo_label = label_map[algo_folder]
            sorted_steps = sorted(step_values.keys())
            original_values = np.array([np.mean(step_values[s]) for s in sorted_steps])
            original_steps = np.array(sorted_steps)
            downsampled_steps, downsampled_values = downsample_data(original_steps, original_values)
            std_values = np.array([
                np.std(original_values[i:i + DOWNSAMPLE_STEP])
                for i in range(0, len(original_values), DOWNSAMPLE_STEP)
            ])
            plot_data[algo_label] = (downsampled_steps, downsampled_values, std_values)
            max_value = max(max_value, np.max(downsampled_values))
            max_step = max(max_step, np.max(downsampled_steps))

        ordered_plot_data = [
            (label, plot_data[label])
            for label in experiment_order
            if label in plot_data
        ]
        if not ordered_plot_data:
            continue

        for idx, (algo_label, (steps, means, stds)) in enumerate(ordered_plot_data):
            color = color_map[algo_label]
            ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2, zorder=1)
            line, = ax.plot(steps, means, color=color, label=algo_label, zorder=2)
            if ax_idx == 0 and algo_label not in legend_labels:
                legend_labels.append(algo_label)
                legend_handles.append(line)

        ax.set_xlabel(x_axis_label, fontproperties=bold_font)
        if ax_idx == 0:
            ax.set_ylabel(y_axis_label, fontproperties=bold_font)

        titles = {
            'model1': "Real-world data (Simple)",
            'model2': "Real-world data (Medium)",
            'model5': "Real-world data (Complicated)"
        }

        ax.set_title(titles[target_model], fontproperties=bold_font, fontsize=12, pad=8)

        ax.set_ylim(bottom=model_lower_bounds[target_model], top=max_value * 1.1)
        ax.set_xlim(0, max_step * 1.1)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6, prune='both'))  # 从4→5
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        ax.xaxis.set_major_formatter(FuncFormatter(format_k))
        ax.yaxis.set_major_formatter(FuncFormatter(format_k))

        for label in ax.get_xticklabels():
            label.set_fontproperties(bold_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(bold_font)

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=True,
        loc="upper center",
        ncol=len(legend_labels),
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=2.0,
        handlelength=3.0,
        fontsize=13,
        prop=bold_font,
        edgecolor='black',
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    pdf_path = "CrossQ_model_comparison.pdf"
    png_path = "CrossQ_model_comparison.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存：\n- {pdf_path}\n- {png_path}")

    plt.show()

if __name__ == "__main__":
    generate_multi_model_plot()