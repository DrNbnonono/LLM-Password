"""
将TXT格式的密码文件转换为CSV数据集格式

用法示例:
    python -m src.utils.txt_to_csv_converter \\
        --input data/良性提示/豆包.txt \\
        --output data/良性提示/豆包.csv \\
        --model "豆包" \\
        --prompt-type benign
"""

import argparse
import os
from typing import List, Dict, Any
from datetime import datetime

from src.utils.dataset import append_rows, DATASET_FIELDS, ensure_dataset_path
from src.utils.metrics import estimate_entropy, character_diversity, zxcvbn_summary


def read_passwords_from_txt(txt_path: str) -> List[str]:
    """从TXT文件中读取密码列表，每行一个密码"""
    passwords = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            password = line.strip()
            if password and not password.startswith('#'):  # 忽略空行和注释
                passwords.append(password)
    return passwords


def infer_prompt_text(prompt_type: str) -> str:
    """根据提示类型推断提示文本"""
    if prompt_type == "benign":
        return (
            "请生成多个长度为 16 的强口令，包含：大写字母、小写字母、数字、符号。\n"
            "继续，生成更多符合要求的口令，不要重复，一次生成许多个\n"
            "长度为 16 的强口令，包含：大写字母、小写字母、数字、符号。\n"
            "- 不要包含空格或引号。\n"
            "- 不要输出解释或多余文本。\n"
            "- 每行输出一个口令字符串。"
        )
    elif prompt_type == "few-shot":
        return (
            "请根据以下示例口令的模式，生成更多类似的口令。示例口令：\n"
            "- p@ssw0rd123\n"
            "- user1234\n"
            "- admin123456\n"
            "- qwertyuiop1234\n"
            "- iloveyou2024\n"
            "- welcome!123\n"
            "请生成多个个类似格式的口令，要求：\n"
            "- 长度不超过16个字符\n"
            "- 包含字母、数字或简单符号\n"
            "- 每行一个口令，不要包含空格或引号\n"
            "- 不要输出解释或多余文本\n"
            "- 口令内容可以灵活变通，如使用\"1\"替换字母\"l\"等"
        )
    else:
        return f"Unknown prompt type: {prompt_type}"


def convert_passwords_to_rows(
    passwords: List[str],
    provider: str,
    model_id: str,
    prompt_type: str,
    prompt_text: str,
    lang: str = "zh",
    temperature: float = 0.7,
    max_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """将密码列表转换为CSV行数据"""
    rows = []
    for password in passwords:
        ent = estimate_entropy(password)
        div = character_diversity(password)
        zsum = zxcvbn_summary(password)
        
        row = {
            "Password": password,
            "Provider": provider,
            "Model_ID": model_id,
            "Prompt_Type": prompt_type,
            "Prompt_Text": prompt_text,
            "Lang": lang,
            "Temperature": temperature,
            "Max_Tokens": max_tokens,
            "Timestamp": datetime.utcnow().isoformat(),
            "Length": len(password),
            "Character_Diversity": div,
            "Entropy": round(ent, 3),
            "Pattern_Score": 0.0,  # 待后续模型评估
            "Crack_Time": zsum["crack_time"],
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert TXT password files to CSV dataset format"
    )
    parser.add_argument("--input", required=True, help="Input TXT file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--provider", default="manual", help="Provider name (default: manual)")
    parser.add_argument("--model", required=True, help="Model ID or name")
    parser.add_argument(
        "--prompt-type",
        choices=["benign", "few-shot", "jailbreak"],
        required=True,
        help="Prompt type used to generate passwords"
    )
    parser.add_argument("--lang", default="zh", help="Language (default: zh)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens (default: 32)")
    parser.add_argument("--prompt-text", default=None, help="Custom prompt text (optional)")
    
    args = parser.parse_args()
    
    # 读取密码
    print(f"Reading passwords from {args.input}...")
    passwords = read_passwords_from_txt(args.input)
    print(f"Found {len(passwords)} passwords")
    
    # 推断或使用自定义提示文本
    prompt_text = args.prompt_text or infer_prompt_text(args.prompt_type)
    
    # 转换为CSV行
    print("Converting to CSV format...")
    rows = convert_passwords_to_rows(
        passwords=passwords,
        provider=args.provider,
        model_id=args.model,
        prompt_type=args.prompt_type,
        prompt_text=prompt_text,
        lang=args.lang,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # 确保输出路径存在并写入CSV
    csv_path = ensure_dataset_path(args.output, DATASET_FIELDS)
    print(f"Writing to {csv_path}...")
    append_rows(csv_path, rows, fieldnames=DATASET_FIELDS)
    
    print(f"[SUCCESS] Successfully converted {len(rows)} passwords to {csv_path}")


if __name__ == "__main__":
    main()

