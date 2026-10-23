"""
检查数据收集进度
"""

import csv
from pathlib import Path
import time


def count_csv_rows(csv_path):
    """计算CSV文件的行数（不含表头）"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # 减去表头
    except:
        return 0


def main():
    project_root = Path(__file__).parent.parent
    fewshot_dir = project_root / "data" / "少样本提示"
    
    # 需要收集的模型
    models = {
        'Hunyuan_MT_7B': {'target': 3000, 'name': 'tencent/Hunyuan-MT-7B'},
        'qwen_2.5_7b_instruct': {'target': 3000, 'name': 'qwen/qwen-2.5-7b-instruct'},
        'gemma_2_9b_it': {'target': 3000, 'name': 'google/gemma-2-9b-it'},
        'glm_4_9b_chat': {'target': 3000, 'name': 'THUDM/glm-4-9b-chat'},
        'Qwen2.5_7B_Instruct': {'target': 3000, 'name': 'Qwen/Qwen2.5-7B-Instruct'},
    }
    
    print("=" * 80)
    print("API模型少样本提示数据收集进度")
    print("=" * 80)
    print(f"\n检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_collected = 0
    total_target = 0
    completed = 0
    
    for model_file, info in models.items():
        csv_path = fewshot_dir / f"{model_file}_fewshot.csv"
        target = info['target']
        total_target += target
        
        if csv_path.exists():
            count = count_csv_rows(csv_path)
            total_collected += count
            percentage = (count / target * 100) if target > 0 else 0
            
            if count >= target:
                status = "✓ 完成"
                completed += 1
            else:
                status = "⏳ 进行中"
            
            print(f"{status} {info['name']}")
            print(f"    进度: {count}/{target} ({percentage:.1f}%)")
            print(f"    文件: {csv_path.name}")
        else:
            print(f"⏳ 等待 {info['name']}")
            print(f"    进度: 0/{target} (0.0%)")
            print(f"    文件: {csv_path.name} (尚未创建)")
        print()
    
    print("=" * 80)
    print(f"总体进度: {completed}/{len(models)} 个模型完成")
    print(f"数据收集: {total_collected}/{total_target} 个密码")
    print(f"完成度: {(total_collected/total_target*100):.1f}%")
    print("=" * 80)
    
    if completed == len(models):
        print("\n✓ 所有数据收集完成！可以进行文件清理。")
    else:
        print(f"\n⏳ 还有 {len(models) - completed} 个模型正在收集数据...")
        print("提示: 数据收集可能需要10-20分钟，请耐心等待")


if __name__ == "__main__":
    main()

