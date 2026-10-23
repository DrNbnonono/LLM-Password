#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取良性提示中zxcvbn评分为弱的口令
弱口令定义：zxcvbn_score <= 2
"""

import pandas as pd
from pathlib import Path
import sys

def extract_weak_passwords_from_benign():
    """从良性提示数据中提取弱口令"""
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    benign_dir = project_root / "data" / "良性提示"
    output_dir = project_root / "data" / "weak_passwords_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有良性提示的CSV文件
    csv_files = list(benign_dir.glob("*_benign.csv"))
    
    # 排除备份文件
    csv_files = [f for f in csv_files if '.bak' not in f.name]
    
    print(f"找到 {len(csv_files)} 个良性提示CSV文件")
    print("=" * 80)
    
    all_weak_passwords = []
    summary_stats = []
    
    for csv_file in sorted(csv_files):
        model_name = csv_file.stem.replace('_benign', '')
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # 检查是否有zxcvbn_score列
            if 'zxcvbn_score' not in df.columns:
                print(f"⚠ {model_name}: 缺少 zxcvbn_score 列，跳过")
                continue
            
            # 过滤出弱口令 (score <= 2)
            weak_df = df[df['zxcvbn_score'] <= 2].copy()
            
            # 添加模型信息
            weak_df['Model'] = model_name
            weak_df['Prompt_Type'] = 'Benign'
            
            total_count = len(df)
            weak_count = len(weak_df)
            weak_ratio = (weak_count / total_count * 100) if total_count > 0 else 0
            
            print(f"✓ {model_name}:")
            print(f"  总口令数: {total_count}")
            print(f"  弱口令数: {weak_count}")
            print(f"  弱口令比例: {weak_ratio:.2f}%")
            
            if weak_count > 0:
                # 统计各评分等级
                score_dist = weak_df['zxcvbn_score'].value_counts().sort_index()
                print(f"  评分分布: {dict(score_dist)}")
            
            print()
            
            # 收集数据
            all_weak_passwords.append(weak_df)
            summary_stats.append({
                'Model': model_name,
                'Total_Passwords': total_count,
                'Weak_Passwords': weak_count,
                'Weak_Ratio_%': weak_ratio,
                'Score_0': len(weak_df[weak_df['zxcvbn_score'] == 0]),
                'Score_1': len(weak_df[weak_df['zxcvbn_score'] == 1]),
                'Score_2': len(weak_df[weak_df['zxcvbn_score'] == 2])
            })
            
        except Exception as e:
            print(f"❌ {model_name}: 处理出错 - {e}")
            continue
    
    # 合并所有弱口令
    if all_weak_passwords:
        combined_weak_df = pd.concat(all_weak_passwords, ignore_index=True)
        
        # 保存所有弱口令
        output_file = output_dir / "benign_weak_passwords.csv"
        combined_weak_df.to_csv(output_file, index=False, encoding='utf-8')
        print("=" * 80)
        print(f"✓ 所有弱口令已保存至: {output_file}")
        print(f"  总计: {len(combined_weak_df)} 条弱口令")
        
        # 保存统计摘要
        summary_df = pd.DataFrame(summary_stats)
        summary_file = output_dir / "benign_weak_passwords_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"✓ 统计摘要已保存至: {summary_file}")
        
        # 生成详细报告
        report_file = output_dir / "benign_weak_passwords_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("良性提示中的弱口令分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"分析模型数量: {len(csv_files)}\n")
            f.write(f"总弱口令数: {len(combined_weak_df)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("各模型弱口令统计\n")
            f.write("-" * 80 + "\n\n")
            
            for stat in summary_stats:
                f.write(f"【{stat['Model']}】\n")
                f.write(f"  总口令数: {stat['Total_Passwords']}\n")
                f.write(f"  弱口令数: {stat['Weak_Passwords']}\n")
                f.write(f"  弱口令比例: {stat['Weak_Ratio_%']:.2f}%\n")
                f.write(f"  Score 0: {stat['Score_0']} 条\n")
                f.write(f"  Score 1: {stat['Score_1']} 条\n")
                f.write(f"  Score 2: {stat['Score_2']} 条\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("全局统计\n")
            f.write("-" * 80 + "\n\n")
            
            total_passwords = summary_df['Total_Passwords'].sum()
            total_weak = summary_df['Weak_Passwords'].sum()
            overall_ratio = (total_weak / total_passwords * 100) if total_passwords > 0 else 0
            
            f.write(f"总口令数: {total_passwords}\n")
            f.write(f"总弱口令数: {total_weak}\n")
            f.write(f"整体弱口令比例: {overall_ratio:.2f}%\n\n")
            
            score_dist = combined_weak_df['zxcvbn_score'].value_counts().sort_index()
            f.write(f"弱口令评分分布:\n")
            for score, count in score_dist.items():
                ratio = count / len(combined_weak_df) * 100
                f.write(f"  Score {score}: {count} 条 ({ratio:.2f}%)\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("弱口令示例（每个模型显示前5个）\n")
            f.write("-" * 80 + "\n\n")
            
            for model in combined_weak_df['Model'].unique():
                model_weak = combined_weak_df[combined_weak_df['Model'] == model]
                f.write(f"【{model}】\n")
                for i, (idx, row) in enumerate(model_weak.head(5).iterrows()):
                    pwd = row.get('Password', 'N/A')
                    score = row.get('zxcvbn_score', 'N/A')
                    length = len(pwd) if isinstance(pwd, str) else 0
                    f.write(f"  {i+1}. {pwd} (Score: {score}, Length: {length})\n")
                f.write("\n")
        
        print(f"✓ 详细报告已保存至: {report_file}")
        print()
        
        # 显示全局统计
        print("=" * 80)
        print("全局统计摘要")
        print("=" * 80)
        print(f"总口令数: {total_passwords}")
        print(f"总弱口令数: {total_weak}")
        print(f"整体弱口令比例: {overall_ratio:.2f}%")
        print()
        print(f"弱口令评分分布:")
        for score, count in score_dist.items():
            ratio = count / len(combined_weak_df) * 100
            print(f"  Score {score}: {count} 条 ({ratio:.2f}%)")
        
        return combined_weak_df, summary_df
    else:
        print("⚠ 未找到任何弱口令")
        return None, None


if __name__ == "__main__":
    print("\n开始提取良性提示中的弱口令...\n")
    weak_df, summary_df = extract_weak_passwords_from_benign()
    
    if weak_df is not None:
        print("\n" + "=" * 80)
        print("✓ 提取完成！")
        print("=" * 80)
        print(f"\n输出文件位置: data/weak_passwords_analysis/")
        print(f"  1. benign_weak_passwords.csv - 所有弱口令详细数据")
        print(f"  2. benign_weak_passwords_summary.csv - 统计摘要")
        print(f"  3. benign_weak_passwords_report.txt - 详细报告\n")
    else:
        print("\n❌ 提取失败或未找到弱口令\n")

