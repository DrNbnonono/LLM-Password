"""
为指定模型收集少样本提示数据

用法:
    python scripts/collect_fewshot_for_model.py --model Gemini --provider openrouter --model-id google/gemini-pro
    python scripts/collect_fewshot_for_model.py --model GPT-5 --provider openrouter --model-id openai/gpt-4
    python scripts/collect_fewshot_for_model.py --model Grok --provider openrouter --model-id x-ai/grok-beta
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="为指定模型收集少样本提示数据"
    )
    parser.add_argument("--model", required=True, help="模型名称（如：Gemini, GPT-5, Grok）")
    parser.add_argument("--provider", choices=["openrouter", "siliconflow"], 
                       default="openrouter", help="API提供商")
    parser.add_argument("--model-id", required=True, 
                       help="API模型ID（如：google/gemini-pro）")
    parser.add_argument("--count", type=int, default=3000, 
                       help="生成密码数量（默认：3000）")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="温度参数（默认：0.7）")
    parser.add_argument("--sleep-ms", type=int, default=500, 
                       help="请求间隔（毫秒，默认：500）")
    
    args = parser.parse_args()
    
    # 构建输出路径
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "少样本提示" / f"{args.model}_fewshot.csv"
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "src.cli.generate",
        "--provider", args.provider,
        "--model", args.model_id,
        "--prompt-type", "few-shot",
        "--count", str(args.count),
        "--temperature", str(args.temperature),
        "--sleep-ms", str(args.sleep_ms),
        "--output", str(output_path),
        "--examples", "p@ssw0rd123;;user1234;;admin123456;;qwertyuiop1234;;iloveyou2024;;welcome!123"
    ]
    
    print("=" * 80)
    print(f"为模型 {args.model} 收集少样本提示数据")
    print("=" * 80)
    print(f"\n提供商: {args.provider}")
    print(f"模型ID: {args.model_id}")
    print(f"数量: {args.count}")
    print(f"输出: {output_path}")
    print(f"\n执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    
    # 执行命令
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print(f"✓ 成功！数据已保存到: {output_path}")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"✗ 失败！错误代码: {e.returncode}")
        print("=" * 80)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()

