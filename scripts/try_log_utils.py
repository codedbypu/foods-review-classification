"""Append structured entries to baseline try-log (experiments/, not outputs/)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import config


def min_class_recall(per_class: dict | None) -> float | None:
    if not per_class:
        return None
    vals = [v for v in per_class.values() if v is not None]
    return min(vals) if vals else None


def min_accuracy(eval_results: dict) -> float | None:
    accs = []
    for v in eval_results.values():
        if "accuracy" in v:
            accs.append(v["accuracy"])
    return min(accs) if accs else None


def format_metrics_table(eval_results: dict) -> str:
    lines = [
        "| set | n | accuracy | f1_macro | mae | min_class_recall |",
        "|-----|---|----------|----------|-----|------------------|",
    ]
    for label in ("wongnai_holdout", "wongnai_test", "mock_test"):
        v = eval_results.get(label, {})
        if "error" in v:
            lines.append(f"| {label} | — | — | — | — | {v['error']} |")
            continue
        mcr = min_class_recall(v.get("per_class_recall"))
        mcr_s = f"{mcr:.4f}" if mcr is not None else "n/a"
        lines.append(
            f"| {label} | {v.get('n', 'n/a')} | {v.get('accuracy', 0):.4f} | "
            f"{v.get('f1_macro', 0):.4f} | {v.get('mae', 0):.4f} | {mcr_s} |"
        )
    return "\n".join(lines)


def append_try_log(
    try_id: str,
    *,
    hypothesis: str,
    config_changes: dict,
    eval_results: dict,
    analysis: str,
    next_step: str,
    val_mlogloss: float | None = None,
    extra: dict | None = None,
) -> None:
    try_log = config.EXPERIMENT_TRY_LOG_PATH
    import os
    from pathlib import Path

    path = Path(try_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    min_acc = min_accuracy(eval_results)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = [
        f"\n## {try_id}\n",
        f"- **Timestamp:** {ts}",
        f"- **สมมติฐาน:** {hypothesis}",
        f"- **การเปลี่ยน config:** `{json.dumps(config_changes, ensure_ascii=False)}`",
    ]
    if val_mlogloss is not None:
        block.append(f"- **val metric (mlogloss/rmse):** {val_mlogloss:.6f}")
    block.append(
        f"- **min_accuracy ทั้ง 3 ชุด:** {min_acc:.4f}" if min_acc else "- **min_accuracy:** n/a"
    )
    block.append("- **Metrics (จาก script):**")
    block.append(format_metrics_table(eval_results))
    block.append(f"- **วิเคราะห์:** {analysis}")
    block.append(f"- **ขั้นถัดไป:** {next_step}")
    if extra:
        block.append(f"- **Extra:** `{json.dumps(extra, ensure_ascii=False)}`")

    mode = "a" if path.is_file() else "w"
    with open(path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Baseline try-log (experiments/baseline/try_log.md)\n\n")
            f.write("โฟลเดอร์นี้แยกจาก `outputs/` — เก็บเฉพาะงานทดลอง/จูน\n")
        f.write("\n".join(block) + "\n")
