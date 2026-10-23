"""
为所有CSV批量添加或更新口令熵(Entropy)列，并显示处理进度。

用法:
  python scripts/add_entropy_to_csv.py --dir data/良性提示
  python scripts/add_entropy_to_csv.py --dir data/少样本提示
  # 或一次性处理两个目录(默认):
  python scripts/add_entropy_to_csv.py
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List

import sys
# 将项目根目录加入sys.path，确保可导入src模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.utils.metrics import estimate_entropy


def list_csv_files(target_dir: Path) -> List[Path]:
    if not target_dir.exists():
        return []
    return sorted([p for p in target_dir.glob("*.csv") if p.is_file()])


def ensure_entropy_header(fieldnames: List[str]) -> List[str]:
    # 确保字段中包含 Entropy 列；若已存在则维持原顺序
    if "Entropy" in fieldnames:
        return fieldnames
    # 将 Entropy 插入到 Length 后(若有)，否则追加到末尾
    if "Length" in fieldnames:
        idx = fieldnames.index("Length") + 1
        return fieldnames[:idx] + ["Entropy"] + fieldnames[idx:]
    return fieldnames + ["Entropy"]


def process_csv_file(csv_path: Path, backup: bool = True) -> None:
    tmp_out = csv_path.with_suffix(".entropy.tmp.csv")
    rows_processed = 0

    # 读取
    with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        in_fields = reader.fieldnames or []
        out_fields = ensure_entropy_header(in_fields.copy())

        # 写出
        with open(tmp_out, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            # 进度条
            buf: List[dict] = list(reader)
            for row in tqdm(buf, desc=f"Entropy: {csv_path.name}"):
                pwd = row.get("Password", "")
                # 重新计算口令熵
                ent = estimate_entropy(pwd)
                row["Entropy"] = round(ent, 3)
                writer.writerow(row)
                rows_processed += 1

    # 备份并原子替换
    if backup:
        bak = csv_path.with_suffix(".bak.csv")
        try:
            if bak.exists():
                bak.unlink()
        except Exception:
            pass
        os.replace(csv_path, bak)
    os.replace(tmp_out, csv_path)

    print(f"✓ {csv_path.name} updated, rows: {rows_processed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch add/update Entropy column for CSV datasets")
    parser.add_argument("--dir", default="", help="Target directory to process (defaults to both data/良性提示 and data/少样本提示)")
    parser.add_argument("--no-backup", action="store_true", help="Do not keep .bak.csv backup of original file")
    args = parser.parse_args()

    target_dirs: List[Path]
    if args.dir:
        target_dirs = [Path(args.dir)]
    else:
        target_dirs = [Path("data") / "良性提示", Path("data") / "少样本提示"]

    all_csvs: List[Path] = []
    for d in target_dirs:
        all_csvs.extend(list_csv_files(d))

    if not all_csvs:
        print("No CSV files found.")
        return

    print(f"Found {len(all_csvs)} CSV files.")
    for csv_path in all_csvs:
        try:
            process_csv_file(csv_path, backup=not args.no_backup)
        except Exception as e:  # noqa: BLE001
            print(f"warning: failed to process {csv_path}: {e}")


if __name__ == "__main__":
    main()
