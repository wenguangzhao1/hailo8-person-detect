#!/usr/bin/env python3
"""
删除指定置信度范围的图片和数据库记录。

用法:
    python3 delete_low_confidence.py --min-conf 0.35 --max-conf 0.40
    python3 delete_low_confidence.py --min-conf 0.0 --max-conf 0.40 --dry-run  # 仅查看不删除
"""

import argparse
import os
import sqlite3

def main():
    parser = argparse.ArgumentParser(description="删除低置信度检测图片")
    parser.add_argument("--output-dir", default="/home/msj/zw/output",
                        help="Output directory path")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="Minimum confidence threshold (default: 0.0)")
    parser.add_argument("--max-conf", type=float, required=True,
                        help="Maximum confidence threshold")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计不删除")
    args = parser.parse_args()

    db_path = os.path.join(args.output_dir, "detections.db")
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 统计匹配的图片数量
    cursor.execute("""
        SELECT DISTINCT filename, confidence
        FROM detections
        WHERE confidence >= ? AND confidence <= ?
        ORDER BY confidence DESC
    """, (args.min_conf, args.max_conf))

    results = cursor.fetchall()
    total = len(results)

    if total == 0:
        print(f"✓ 没有找到置信度在 {args.min_conf:.2f}-{args.max_conf:.2f} 范围的图片")
        conn.close()
        return

    print(f"\n📊 找到 {total} 张图片 置信度范围 {args.min_conf:.2f}-{args.max_conf:.2f}")
    print(f"   置信度分布:")
    conf_buckets = {}
    for filename, conf in results:
        bucket = f"{conf:.2f}"
        conf_buckets[bucket] = conf_buckets.get(bucket, 0) + 1
    for bucket, count in sorted(conf_buckets.items()):
        print(f"   {bucket}: {count} 张")

    if args.dry_run:
        print(f"\n🔍 [DRY-RUN] 示例文件名:")
        for filename, conf in results[:10]:
            print(f"   {filename} (conf: {conf:.3f})")
        if total > 10:
            print(f"   ... 还有 {total-10} 张")
        conn.close()
        return

    # 确认删除
    print(f"\n⚠️  即将删除 {total} 张图片及其数据库记录")
    response = input("确认删除? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ 取消删除")
        conn.close()
        return

    # 删除图片文件
    deleted_files = 0
    for filename, _ in results:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                deleted_files += 1
            except Exception as e:
                print(f"❌ 删除失败 {filename}: {e}")

    # 删除数据库记录
    cursor.execute("""
        DELETE FROM detections
        WHERE confidence >= ? AND confidence <= ?
    """, (args.min_conf, args.max_conf))
    deleted_records = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"\n✅ 删除完成:")
    print(f"   图片文件: {deleted_files} 张")
    print(f"   数据库记录: {deleted_records} 条")

if __name__ == "__main__":
    main()