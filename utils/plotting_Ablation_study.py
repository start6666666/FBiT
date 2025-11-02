import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties


label_map = {
    'SAC': 'SAC',
    'TFP': 'TFP',
    'TFP+BIFN': 'TFP-BIFN',
    'FBiT': 'FBiT'
}

experiment_order = [
    'SAC',
    'TFP',
    'TFP-BIFN',
    'FBiT',
]

palette = "muted"
x_axis_label = 'Algorithm'
y_axis_label = 'Cumulative Return'


target_models = ['model1', 'model2', 'model3']
title = ['Simple', 'Medium', 'Complicated']
learn = 'learn1'
root_dir = 'results/SupplyChainGReal-v0/669'


bold_font = FontProperties(weight='bold')



def format_k(x, pos):
    if x == 0:
        return "0"
    return f"{x/1000:.1f}k" if (x % 1000) != 0 else f"{int(x/1000)}k"


def load_model_data(model):

    value_column = 'Return'
    model_data = []
    for dirpath, _, filenames in os.walk(root_dir):
        path_parts = dirpath.split(os.sep)
        if (model in path_parts and
            learn in path_parts and
            'test_results.csv' in filenames):
            
            csv_path = os.path.join(dirpath, 'test_results.csv')
            try:
                df = pd.read_csv(csv_path)
                if model in path_parts:
                    model_idx = path_parts.index(model)
                    if model_idx > 0:
                        experiment_label = path_parts[model_idx - 1]
                        df['experiment_raw'] = experiment_label
                        model_data.append(df)
            except Exception as e:
                print(f"处理文件失败 {csv_path}: {e}")
    
    if not model_data:
        print(f"警告：未找到 {model} 的数据")
        return None
    
    combined_df = pd.concat(model_data, ignore_index=True)
    if value_column not in combined_df.columns:
        print(f"错误：{model} 缺少 '{value_column}' 列")
        return None
    
    combined_df[x_axis_label] = combined_df['experiment_raw'].map(label_map)
    combined_df = combined_df.dropna(subset=[x_axis_label])
    return combined_df


def generate_multi_model_barplot():
    model_dfs = {model: load_model_data(model) for model in target_models}
    valid_models = [m for m in target_models if model_dfs[m] is not None]
    if not valid_models:
        print("错误：所有模型均无有效数据")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 11,
        "figure.dpi": 300,
        "lines.linewidth": 1.2,
        "errorbar.capsize": 3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=False)

    plt.subplots_adjust(wspace=0.3)


    for ax_idx, model in enumerate(valid_models):
        ax = axes[ax_idx] if len(valid_models) > 1 else axes
        df = model_dfs[model]

        sns.barplot(
            x=x_axis_label,
            y='Return',
            data=df,
            order=experiment_order,
            palette=palette,
            ax=ax,
            width=0.5,
            errorbar='sd',
            capsize=0.1
        )


        ax.set_title(title[ax_idx], fontproperties=bold_font, pad=8)


        ax.set_xlabel(x_axis_label, fontproperties=bold_font, labelpad=5)
        if ax_idx == 0:
            ax.set_ylabel(y_axis_label, fontproperties=bold_font, labelpad=5)
        else:
            ax.set_ylabel('')


        ax.yaxis.set_major_formatter(FuncFormatter(format_k))

        for label in ax.get_xticklabels():
            label.set_fontproperties(bold_font)
            label.set_rotation(0)
        for label in ax.get_yticklabels():
            label.set_fontproperties(bold_font)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.5)

    plt.tight_layout()

    pdf_path = "multi_model_final_performance.pdf"
    png_path = "multi_model_final_performance.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"图表已保存：\n- {pdf_path}\n- {png_path}")

    plt.show()


if __name__ == '__main__':
    generate_multi_model_barplot()