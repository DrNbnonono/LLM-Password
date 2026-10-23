"""
自动按模型分割API CSV文件（不需要交互确认）
"""

import csv
from pathlib import Path
from collections import defaultdict


def split_csv_by_model(input_csv, output_dir):
    """将CSV文件按Model_ID分组并保存为单独的文件"""
    
    model_data = defaultdict(list)
    fieldnames = None
    
    print(f"\n处理文件: {input_csv.name}")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            model_id = row.get('Model_ID', 'unknown')
            model_data[model_id].append(row)
    
    print(f"  发现 {len(model_data)} 个模型:")
    for model_id, rows in model_data.items():
        print(f"    - {model_id}: {len(rows)} 条")
    
    created_files = {}
    
    for model_id, rows in model_data.items():
        # 生成文件名
        model_name = model_id.split('/')[-1]
        model_name = model_name.replace('-', '_').replace('.', '_')
        output_file = output_dir / f"{model_name}_benign.csv"
        
        # 检查文件是否存在
        if output_file.exists():
            print(f"  ⚠ 文件已存在，将覆盖: {output_file.name}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        created_files[model_id] = output_file
        print(f"  ✓ 创建: {output_file.name} ({len(rows)} 条)")
    
    return created_files


def main():
    project_root = Path(__file__).parent.parent
    benign_dir = project_root / "data" / "良性提示"
    
    files_to_split = [
        benign_dir / "llm_passwords.csv",
        benign_dir / "llm_passwords_v2.csv",
        benign_dir / "llm_passwords_GLM_9B.csv",
    ]
    
    print("=" * 80)
    print("按模型分割API收集的CSV文件")
    print("=" * 80)
    
    all_created = {}
    files_processed = []
    
    for csv_file in files_to_split:
        if csv_file.exists():
            created = split_csv_by_model(csv_file, benign_dir)
            all_created.update(created)
            files_processed.append(csv_file)
        else:
            print(f"\n⚠ 跳过（文件不存在）: {csv_file.name}")
    
    print("\n" + "=" * 80)
    print("分割结果汇总")
    print("=" * 80)
    
    print(f"\n共创建 {len(all_created)} 个模型文件:")
    for model_id, output_file in sorted(all_created.items()):
        if output_file.exists():
            size_kb = output_file.stat().st_size / 1024
            with open(output_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f) - 1
            print(f"  - {output_file.name}")
            print(f"    模型: {model_id}")
            print(f"    数据: {line_count} 条 ({size_kb:.1f} KB)")
    
    # 删除空的CSV文件
    print("\n删除空文件:")
    empty_files = [
        benign_dir / "llm_passwords_gpt_oss.csv",
        benign_dir / "llm_passwords_phi.csv",
    ]
    
    for empty_file in empty_files:
        if empty_file.exists() and empty_file.stat().st_size < 200:
            empty_file.unlink()
            print(f"  ✓ 删除: {empty_file.name}")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print("\n原始文件保留在:")
    for f in files_processed:
        print(f"  - {f.name}")
    print("\n如需删除原始文件，请手动删除或运行:")
    print("  del data\\良性提示\\llm_passwords*.csv")


if __name__ == "__main__":
    main()

