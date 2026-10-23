#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为所有CSV数据集添加zxcvbn评估结果列
在原有CSV基础上添加：zxcvbn_score, zxcvbn_guesses_log10, zxcvbn_crack_time 等列
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from zxcvbn import zxcvbn
except ImportError:
    print("❌ 错误: 未安装 zxcvbn 库")
    print("请运行: pip install zxcvbn")
    sys.exit(1)


def safe_zxcvbn(password):
    """安全地调用zxcvbn，处理异常情况"""
    try:
        if pd.isna(password) or password == '':
            return {
                'score': 0,
                'guesses': 1,
                'guesses_log10': 0.0,
                'crack_times_seconds': {'offline_slow_hashing_1e4_per_second': 0.0001},
                'crack_times_display': {'offline_slow_hashing_1e4_per_second': '< 1秒'}
            }
        
        password_str = str(password)
        
        # zxcvbn 限制最大72字符
        if len(password_str) > 72:
            result = zxcvbn(password_str[:72])
        else:
            result = zxcvbn(password_str)
        
        return result
        
    except Exception as e:
        print(f"    ⚠ zxcvbn评估异常 (密码: {password[:20]}...): {e}")
        return {
            'score': 0,
            'guesses': 1,
            'guesses_log10': 0.0,
            'crack_times_seconds': {'offline_slow_hashing_1e4_per_second': 0.0001},
            'crack_times_display': {'offline_slow_hashing_1e4_per_second': '< 1秒'}
        }


def add_zxcvbn_columns(csv_path):
    """为单个CSV文件添加zxcvbn评估列"""
    try:
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        # 检查必要列
        if 'Password' not in df.columns:
            print(f"  ⚠ 跳过 (无Password列)")
            return False
        
        # 检查是否已有zxcvbn列
        if 'zxcvbn_score' in df.columns:
            print(f"  ℹ 已存在zxcvbn列，将重新计算")
        
        # 创建备份
        backup_path = csv_path.parent / f"{csv_path.stem}.bak.csv"
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)
            print(f"  ✓ 已创建备份: {backup_path.name}")
        
        # 初始化zxcvbn结果列
        df['zxcvbn_score'] = 0
        df['zxcvbn_guesses_log10'] = 0.0
        df['zxcvbn_log2_guesses'] = 0.0  # 等效熵
        df['zxcvbn_crack_time_display'] = ''
        df['zxcvbn_crack_time_seconds'] = 0.0
        
        # 逐行评估
        print(f"  开始评估 {len(df)} 条口令...")
        for idx in tqdm(df.index, desc=f"  处理 {csv_path.name}", leave=False):
            password = df.at[idx, 'Password']
            result = safe_zxcvbn(password)
            
            df.at[idx, 'zxcvbn_score'] = int(result['score'])
            df.at[idx, 'zxcvbn_guesses_log10'] = float(result['guesses_log10'])
            
            # 计算 log2(guesses) 作为等效熵
            guesses = float(result['guesses'])
            if guesses > 0:
                df.at[idx, 'zxcvbn_log2_guesses'] = np.log2(guesses)
            else:
                df.at[idx, 'zxcvbn_log2_guesses'] = 0.0
            
            df.at[idx, 'zxcvbn_crack_time_display'] = \
                result['crack_times_display']['offline_slow_hashing_1e4_per_second']
            df.at[idx, 'zxcvbn_crack_time_seconds'] = \
                float(result['crack_times_seconds']['offline_slow_hashing_1e4_per_second'])
        
        # 保存更新后的CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 已保存: {csv_path.name}")
        
        # 显示统计
        weak_count = (df['zxcvbn_score'] <= 2).sum()
        weak_ratio = weak_count / len(df) * 100
        print(f"  📊 弱口令比例 (score≤2): {weak_count}/{len(df)} ({weak_ratio:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "批量添加 zxcvbn 评估列到 CSV" + " " * 20 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # 统计
    total_files = 0
    success_files = 0
    
    # 遍历良性提示和少样本提示目录
    for prompt_type_dir in ["良性提示", "少样本提示"]:
        dir_path = data_dir / prompt_type_dir
        
        if not dir_path.exists():
            print(f"⚠ 目录不存在: {dir_path}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"处理目录: {prompt_type_dir}")
        print(f"{'=' * 80}\n")
        
        # 遍历所有CSV文件
        csv_files = sorted(dir_path.glob("*.csv"))
        # 跳过备份文件
        csv_files = [f for f in csv_files if ".bak" not in f.name]
        
        for csv_file in csv_files:
            print(f"\n📄 {csv_file.name}")
            total_files += 1
            
            if add_zxcvbn_columns(csv_file):
                success_files += 1
    
    # 最终统计
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 30 + "处理完成!" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print(f"总计处理: {total_files} 个文件")
    print(f"成功: {success_files} 个")
    print(f"失败: {total_files - success_files} 个")
    
    if success_files == total_files:
        print("\n✅ 所有文件已成功添加 zxcvbn 评估列!")
    else:
        print(f"\n⚠ 部分文件处理失败，请检查错误信息")


if __name__ == "__main__":
    main()

