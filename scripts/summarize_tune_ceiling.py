"""Append ceiling analysis from tune log to experiments try-log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

LOG_JSON = Path(config.EXPERIMENT_TUNE_LOG_PATH)
TRY_LOG = Path(config.EXPERIMENT_TRY_LOG_PATH)
TARGET = 0.9


def main() -> None:
    with open(LOG_JSON, encoding="utf-8") as f:
        rows = json.load(f)

    best = max(rows, key=lambda r: r.get("min_accuracy") or 0)
    passed = [r["name"] for r in rows if r.get("passed_target")]

    lines = [
        "\n## Ceiling analysis (หลังทดลอง 19 configs)\n",
        f"- **เป้าหมาย:** min accuracy > {TARGET} บน holdout + HF test + mock",
        f"- **ผล:** ไม่มี config ผ่าน ({len(passed)} passed)",
        f"- **Best by min_accuracy:** `{best['name']}` → min_acc={best['min_accuracy']:.4f}",
        "",
        "| name | min_acc | holdout | HF test | mock |",
        "|------|---------|---------|---------|------|",
    ]
    for r in sorted(rows, key=lambda x: -(x.get("min_accuracy") or 0))[:8]:
        ev = r["eval"]
        lines.append(
            f"| {r['name']} | {r.get('min_accuracy', 0):.4f} | "
            f"{ev.get('wongnai_holdout', {}).get('accuracy', 0):.4f} | "
            f"{ev.get('wongnai_test', {}).get('accuracy', 0):.4f} | "
            f"{ev.get('mock_test', {}).get('accuracy', 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "### วินิจฉัย",
            "- **HF test** (balanced 20%/class) ติดที่ ~0.25–0.31 — train เอียง 4★ (~41%)",
            "  ทำให้โมเดลทาย 4★ มาก; recall ดาว 1–2 ยังต่ำ (<10%)",
            "- **Holdout** สูงสุด ~0.47 — ชนะ majority (~0.41) แต่ห่างจาก 0.9",
            "- **Mock test** ถึง 1.0 เมื่อ `BASELINE_MOCK_MIX_FRACTION>=0.1`",
            "  แต่ไม่ดึง HF/holdout ขึ้นถึง 0.9 (domain gap)",
            "- TF-IDF + XGB **ceiling ~0.31 min_acc** บนชุดนี้; ต้อง transformer / เปลี่ยน task",
            "  หรือผ่อนเป้า eval ถ้าต้องการ accuracy สูง",
            "",
            "### Config ที่บันทึกใน config.py",
            f"- Winner: `{best['name']}` — mock_mix=0.2, char TF-IDF, oversample×5,",
            "  undersample 4★ 65%, boost=3.0",
            f"- รายละเอียด: `{LOG_JSON}`",
        ]
    )

    TRY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRY_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended ceiling report to {TRY_LOG}")


if __name__ == "__main__":
    main()
