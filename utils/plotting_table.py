import pandas as pd
import os

root_directory = "results/SupplyChainGReal-v0/669/"
output_file = os.path.join(root_directory, "return_summary_2025.10.23.csv")

summary_df = pd.DataFrame(columns=["file_path", "Return_mean±std"])

for root, dirs, files in os.walk(root_directory):
    for filename in files:

        if filename.startswith("test_results") and filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            try:

                df = pd.read_csv(file_path)
                
                if "Return" not in df.columns:
                    print(f"⚠️ {file_path} 中无'Return'列，跳过处理")
                    continue
                
                return_series = df["Return"]
                if not pd.api.types.is_numeric_dtype(return_series):
                    print(f"⚠️ {file_path} 的'Return'列不是数值类型，跳过处理")
                    continue
                
                mean = round(return_series.mean(), 2)
                std = round(return_series.std(), 2)
                return_stats = f"{mean}±{std}"
                
                summary_df = pd.concat(
                    [summary_df, 
                     pd.DataFrame([[file_path, return_stats]], 
                                  columns=["file_path", "Return_mean±std"])],
                    ignore_index=True
                )
                print(f"✅ 已处理：{file_path} 的Return列")
                
            except Exception as e:
                print(f"❌ 处理{file_path}时出错：{str(e)}")

if not summary_df.empty:
    summary_df.to_csv(output_file, index=False)
    print(f"\n🎉 所有Return列结果已汇总保存至：{output_file}")
else:
    print("\n⚠️ 未找到含有效'Return'列的文件，未生成汇总结果")