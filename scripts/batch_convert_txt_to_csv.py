"""
批量将data目录下的TXT文件转换为CSV格式

用法:
    python scripts/batch_convert_txt_to_csv.py
"""

import os
import subprocess
from pathlib import Path


def convert_txt_to_csv(txt_path: str, csv_path: str, model: str, prompt_type: str):
    """调用转换工具将单个TXT文件转换为CSV"""
    cmd = [
        "python", "-m", "src.utils.txt_to_csv_converter",
        "--input", txt_path,
        "--output", csv_path,
        "--model", model,
        "--prompt-type", prompt_type,
    ]
    
    print(f"\n转换: {txt_path} -> {csv_path}")
    try:
        # 在Windows上使用系统默认编码，避免subprocess编码问题
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                              encoding='utf-8', errors='replace')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"错误: 转换失败")
        if e.stderr:
            print(e.stderr)
        raise


def main():
    project_root = Path(__file__).parent.parent
    
    # 定义需要转换的文件
    conversions = []
    
    # 良性提示的TXT文件
    benign_dir = project_root / "data" / "良性提示"
    benign_txts = [
        "deepseek.txt",
        "Gemini.txt",
        "GLM.txt",
        "GPT-5.txt",
        "Grok.txt",
        "kimi.txt",
        "minmax.txt",
        "豆包.txt",
        "通义千问.txt",
    ]
    
    for txt_file in benign_txts:
        txt_path = benign_dir / txt_file
        if txt_path.exists():
            model_name = txt_file.replace(".txt", "")
            csv_file = f"{model_name}_benign.csv"
            csv_path = benign_dir / csv_file
            conversions.append({
                "txt": str(txt_path),
                "csv": str(csv_path),
                "model": model_name,
                "prompt_type": "benign"
            })
    
    # 少样本提示的TXT文件
    fewshot_dir = project_root / "data" / "少样本提示"
    fewshot_txts = [
        "deepseek-R1.txt",
        "deepseek.txt",
        "GLM.txt",
        "kimi.txt",
        "minmax.txt",
        "豆包.txt",
        "通义千问.txt",
    ]
    
    for txt_file in fewshot_txts:
        txt_path = fewshot_dir / txt_file
        if txt_path.exists():
            model_name = txt_file.replace(".txt", "")
            csv_file = f"{model_name}_fewshot.csv"
            csv_path = fewshot_dir / csv_file
            conversions.append({
                "txt": str(txt_path),
                "csv": str(csv_path),
                "model": model_name,
                "prompt_type": "few-shot"
            })
    
    print(f"找到 {len(conversions)} 个文件需要转换")
    print("=" * 80)
    
    # 执行转换
    success_count = 0
    fail_count = 0
    
    for i, conv in enumerate(conversions, 1):
        print(f"\n[{i}/{len(conversions)}] 处理: {conv['model']} ({conv['prompt_type']})")
        try:
            convert_txt_to_csv(
                txt_path=conv["txt"],
                csv_path=conv["csv"],
                model=conv["model"],
                prompt_type=conv["prompt_type"]
            )
            success_count += 1
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            fail_count += 1
    
    print("\n" + "=" * 80)
    print(f"\n转换完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(conversions)}")


if __name__ == "__main__":
    main()

