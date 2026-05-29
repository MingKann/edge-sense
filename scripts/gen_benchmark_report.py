"""生成 Benchmark 报告（供 CI 调用）"""
import json, sys

with open(".benchmark_results.json") as f:
    data = json.load(f)

rows = []
for b in data.get("benchmarks", []):
    name = b["name"].replace("test_bench_", "").replace("_", " ")
    mean_us = b["stats"]["mean"] * 1e6
    rows.append((mean_us, name, b["stats"]))

rows.sort()

print("## 性能基准报告")
print()
print("| 操作 | 平均耗时 | 最小值 | 最大值 | 样本数 |")
print("|---|---|---|---|---|")
for mean_us, name, stats in rows:
    print(
        f"| {name} | {mean_us:.2f} µs | {stats['min']*1e6:.2f} µs | "
        f"{stats['max']*1e6:.2f} µs | {stats['rounds']} |"
    )
print()
print(f"*生成时间: {data['datetime']}*")
