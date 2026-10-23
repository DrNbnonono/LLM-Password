"""
清理项目中的冗余文件
"""

from pathlib import Path
import os


def main():
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("清理冗余文件")
    print("=" * 80)
    
    # 要删除的文件列表
    files_to_delete = []
    
    # 1. 空的API CSV文件
    benign_dir = project_root / "data" / "良性提示"
    empty_csv_files = [
        benign_dir / "llm_passwords_gpt_oss.csv",
        benign_dir / "llm_passwords_phi.csv",
    ]
    
    for f in empty_csv_files:
        if f.exists() and f.stat().st_size < 200:  # 小于200字节认为是空的
            files_to_delete.append(f)
    
    # 2. __pycache__ 目录
    pycache_dirs = list(project_root.rglob("__pycache__"))
    
    # 显示要删除的文件
    print("\n【文件】")
    if files_to_delete:
        for f in files_to_delete:
            size = f.stat().st_size
            print(f"  - {f.relative_to(project_root)} ({size} bytes)")
    else:
        print("  (无)")
    
    print("\n【__pycache__ 目录】")
    if pycache_dirs:
        for d in pycache_dirs:
            print(f"  - {d.relative_to(project_root)}/")
    else:
        print("  (无)")
    
    if not files_to_delete and not pycache_dirs:
        print("\n✓ 没有冗余文件需要清理")
        return
    
    # 询问确认
    print("\n" + "=" * 80)
    print("是否删除以上文件和目录？")
    print("=" * 80)
    response = input("输入 'yes' 确认删除: ").strip().lower()
    
    if response != 'yes':
        print("\n取消删除操作")
        return
    
    # 执行删除
    deleted_count = 0
    
    for f in files_to_delete:
        try:
            f.unlink()
            print(f"✓ 已删除: {f.relative_to(project_root)}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ 删除失败: {f.relative_to(project_root)} - {e}")
    
    for d in pycache_dirs:
        try:
            import shutil
            shutil.rmtree(d)
            print(f"✓ 已删除: {d.relative_to(project_root)}/")
            deleted_count += 1
        except Exception as e:
            print(f"✗ 删除失败: {d.relative_to(project_root)}/ - {e}")
    
    print(f"\n✓ 清理完成，共删除 {deleted_count} 项")


if __name__ == "__main__":
    main()

