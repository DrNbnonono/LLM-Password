"""
检查哪些模型在少样本提示中缺失数据

更简单直接的分析：比较TXT文件
"""

import os
from pathlib import Path
from typing import Set


def get_model_names_from_txt(directory: Path) -> Set[str]:
    """从TXT文件中提取模型名称"""
    models = set()
    if not directory.exists():
        return models
    
    for file in directory.glob("*.txt"):
        # 直接使用文件名（去掉.txt）
        models.add(file.stem)
    
    return models


def main():
    project_root = Path(__file__).parent.parent
    benign_dir = project_root / "data" / "良性提示"
    fewshot_dir = project_root / "data" / "少样本提示"
    
    benign_models = get_model_names_from_txt(benign_dir)
    fewshot_models = get_model_names_from_txt(fewshot_dir)
    
    print("=" * 80)
    print("少样本提示数据缺失分析")
    print("=" * 80)
    
    print(f"\n良性提示中的模型 ({len(benign_models)}):")
    for model in sorted(benign_models):
        status = "✓" if model in fewshot_models else "✗"
        print(f"  {status} {model}")
    
    missing = benign_models - fewshot_models
    extra = fewshot_models - benign_models
    
    if missing:
        print(f"\n需要收集少样本提示数据的模型 ({len(missing)}):")
        for model in sorted(missing):
            print(f"  • {model}")
        
        print("\n" + "=" * 80)
        print("收集建议")
        print("=" * 80)
        print("\n请为以下模型手动收集少样本提示数据，或通过API自动收集：")
        print("\n使用的少样本提示prompt:")
        print("-" * 80)
        print("""请根据以下示例口令的模式，生成更多类似的口令。示例口令：
- p@ssw0rd123
- user1234
- admin123456
- qwertyuiop1234
- iloveyou2024
- welcome!123
请生成多个个类似格式的口令，要求：
- 长度不超过16个字符
- 包含字母、数字或简单符号
- 每行一个口令，不要包含空格或引号
- 不要输出解释或多余文本
- 口令内容可以灵活变通，如使用"1"替换字母"l"等""")
        print("-" * 80)
        
        for model in sorted(missing):
            print(f"\n模型: {model}")
            print(f"  1. 手动收集: 将结果保存到 data/少样本提示/{model}.txt")
            print(f"  2. 自动转换: python -m src.utils.txt_to_csv_converter \\")
            print(f"                --input data/少样本提示/{model}.txt \\")
            print(f"                --output data/少样本提示/{model}_fewshot.csv \\")
            print(f"                --model {model} --prompt-type few-shot")
    else:
        print("\n✓ 所有良性提示的模型在少样本提示中都有数据！")
    
    if extra:
        print(f"\n额外的模型（仅在少样本提示中）({len(extra)}):")
        for model in sorted(extra):
            print(f"  + {model}")
    
    print("\n" + "=" * 80)
    
    # 统计
    print(f"\n数据完整性: {len(benign_models - missing)}/{len(benign_models)} " +
          f"({100 * (1 - len(missing) / max(len(benign_models), 1)):.1f}%)")


if __name__ == "__main__":
    main()

