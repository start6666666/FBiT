import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

label_map = {
    'SAC': 'SAC',
    'TFP+BIFN-parallel.py': 'TFP-BIFN-P',
    'TFP+BIFN.py': 'TFP-BIFN-S',
    'FBiT-parallel': 'FBiT-P',
    'FBiT': 'FBiT-S'
}

experiment_order = [
    'SAC',
    'TFP-BIFN-P',
    'TFP-BIFN-S',
    'FBiT-P',
    'FBiT-S'
]

palette = "muted"
x_axis_label = 'Algorithm'
y_axis_label = 'Cumulative Return'

mode = 'model6'
learn = 'learn1'
root_dir = 'results/SupplyChainG-v0/7548'
chart_title = "Simulate Env (Mode 6)"

def format_k(x, pos):
    if x == 0:
        return "0"
    return f"{x/1000:.1f}K" if (x % 1000) != 0 else f"{int(x/1000)}K"

def generate_publishable_plot():
    value_column = 'Return'
    all_data = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        path_parts = dirpath.split(os.sep)
        if (mode in path_parts and
            learn in path_parts and
            'test_results.csv' in filenames and
            '改成transformer只接受需求' not in path_parts):
            
            csv_path = os.path.join(dirpath, 'test_results.csv')
            try:
                df = pd.read_csv(csv_path)
                model_index = path_parts.index(mode)
                if model_index > 0:
                    experiment_label = path_parts[model_index - 1]
                    df['experiment_raw'] = experiment_label
                    all_data.append(df)
                    print(f"成功加载: {csv_path}")
            except Exception as e:
                print(f"处理文件失败 {csv_path}: {e}")

    if not all_data:
        print("错误: 未找到符合条件的数据文件。")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    if value_column not in combined_df.columns:
        print(f"错误: 指定的列 '{value_column}' 不存在。")
        return

    combined_df[x_axis_label] = combined_df['experiment_raw'].map(label_map)
    combined_df.dropna(subset=[x_axis_label], inplace=True)


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.labelsize'] = 6    
    plt.rcParams['xtick.labelsize'] = 6   
    plt.rcParams['ytick.labelsize'] = 6   
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 2

    fig, ax = plt.subplots(figsize=(3.2, 2))

    sns.boxplot(
        x=x_axis_label,
        y=value_column,
        data=combined_df,
        order=experiment_order,
        palette=palette,
        hue=x_axis_label,
        legend=False,
        width=0.4,
        linewidth=0.7,
        ax=ax
    )


    ax.set_xlabel(x_axis_label, fontweight='bold', labelpad=3)
    ax.set_ylabel(y_axis_label, fontweight='bold', labelpad=3)

    plt.xticks(rotation=0, ha='center',fontweight='bold', fontsize=5.5)

    ax.yaxis.set_major_formatter(FuncFormatter(format_k))

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.4)
    ax.grid(axis='x', alpha=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.3)

    plt.savefig('comparison_plot_single_col.pdf', format='pdf', 
                bbox_inches='tight', pad_inches=0.03)
    plt.savefig('comparison_plot_single_col.png', dpi=300, 
                bbox_inches='tight', pad_inches=0.03)

    print("\n单栏图表已保存（适配双栏论文）")
    plt.show()


if __name__ == '__main__':
    generate_publishable_plot()