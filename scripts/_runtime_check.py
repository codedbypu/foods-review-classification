"""
One-off runtime checker — imports modules, runs CLI smoke, writes docs/RUNTIME_CHECK_REPORT.md
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "wongnai-restaurant-review_train.csv"
SMOKE_PARQUET = ROOT / "data" / "_runtime_check_processed.parquet"
SMOKE_ARTIFACT = ROOT / "artifacts" / "_runtime_check_baseline"
SMOKE_REPORT = ROOT / "reports" / "_runtime_check_metrics.json"
SMOKE_OUT = ROOT / "outputs" / "_runtime_check_scored.parquet"

RRIS_MODULES = [
    "rris",
    "rris.config",
    "rris.logging_utils",
    "rris.progress",
    "rris.data.io",
    "rris.data.text",
    "rris.data.tokenizers",
    "rris.data.datasets",
    "rris.data.aspects",
    "rris.integrity.anomaly",
    "rris.viz.colors",
    "rris.viz.geo_export",
    "rris.models.baseline.tfidf_xgb",
    "rris.models.baseline.xgb_device",
    "rris.models.xlmr.sentiment_trainer",
    "rris.models.xlmr.aspect_extractor",
]

CLI_SCRIPTS = [
    "preprocess.py",
    "train_baseline_xgb.py",
    "train_xlmr_sentiment.py",
    "evaluate.py",
    "score_and_flag.py",
]

PIPELINE_STEPS = [
    (
        "preprocess",
        [
            "scripts/preprocess.py",
            "--input",
            str(DATA_CSV),
            "--out",
            str(SMOKE_PARQUET),
            "--no_progress",
        ],
    ),
    (
        "train_baseline_xgb",
        [
            "scripts/train_baseline_xgb.py",
            "--input",
            str(SMOKE_PARQUET),
            "--out_dir",
            str(SMOKE_ARTIFACT),
            "--max_rows",
            "500",
            "--device",
            "cpu",
            "--n_jobs",
            "2",
            "--no_progress",
        ],
    ),
    (
        "evaluate_baseline",
        [
            "scripts/evaluate.py",
            "--input",
            str(SMOKE_PARQUET),
            "--model_type",
            "baseline_xgb",
            "--artifact_dir",
            str(SMOKE_ARTIFACT),
            "--out",
            str(SMOKE_REPORT),
            "--n_jobs",
            "2",
            "--no_progress",
        ],
    ),
    (
        "score_and_flag",
        [
            "scripts/score_and_flag.py",
            "--input",
            str(SMOKE_PARQUET),
            "--model_type",
            "baseline_xgb",
            "--artifact_dir",
            str(SMOKE_ARTIFACT),
            "--out",
            str(SMOKE_OUT),
            "--skip_aspects",
            "--n_jobs",
            "2",
            "--no_progress",
        ],
    ),
]

XLMR_SMOKE = [
    "scripts/train_xlmr_sentiment.py",
    "--input",
    str(SMOKE_PARQUET),
    "--out_dir",
    str(ROOT / "artifacts" / "_runtime_check_xlmr"),
    "--epochs",
    "1",
    "--train_batch_size",
    "4",
    "--eval_batch_size",
    "4",
    "--no_progress",
]


def run_subprocess(args: list[str], timeout: int | None = 3600) -> dict:
    cmd = [sys.executable, *[str(a) for a in args]]
    try:
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout_tail": (r.stdout or "")[-4000:],
            "stderr_tail": (r.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout_tail": (e.stdout or "")[-2000:] if e.stdout else "",
            "stderr_tail": f"TIMEOUT after {timeout}s",
        }
    except Exception as e:
        return {"ok": False, "returncode": -1, "stderr_tail": traceback.format_exc()}


def test_imports() -> list[dict]:
    sys.path.insert(0, str(ROOT / "src"))
    rows = []
    for mod in RRIS_MODULES:
        try:
            importlib.import_module(mod)
            rows.append({"module": mod, "ok": True, "error": None})
        except Exception as e:
            rows.append({"module": mod, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return rows


def test_help(script: str) -> dict:
    return run_subprocess([f"scripts/{script}", "--help"], timeout=120)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Runtime Check Report",
        "",
        f"Generated: **{ts}**",
        f"Python: `{sys.version.split()[0]}` — `{sys.executable}`",
        f"Repo root: `{ROOT}`",
        "",
    ]

    if not DATA_CSV.exists():
        lines.extend(
            [
                "## Blocker",
                "",
                f"Missing input data: `{DATA_CSV}` — pipeline smoke skipped.",
                "",
            ]
        )
        (ROOT / "docs" / "RUNTIME_CHECK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        print("Wrote docs/RUNTIME_CHECK_REPORT.md (no data file)")
        return

    # Imports
    lines.append("## 1. Module imports (`src/rris`)")
    lines.append("")
    lines.append("| Module | Status | Error |")
    lines.append("|--------|--------|-------|")
    import_rows = test_imports()
    for row in import_rows:
        st = "PASS" if row["ok"] else "**FAIL**"
        err = row["error"] or "—"
        lines.append(f"| `{row['module']}` | {st} | {err} |")
    lines.append("")

    # CLI --help
    lines.append("## 2. CLI `--help`")
    lines.append("")
    lines.append("| Script | Status | returncode | stderr (tail) |")
    lines.append("|--------|--------|------------|---------------|")
    for script in CLI_SCRIPTS:
        r = test_help(script)
        st = "PASS" if r["ok"] else "**FAIL**"
        err = (r.get("stderr_tail") or "—").replace("\n", " ").replace("|", "\\|")[:200]
        lines.append(f"| `{script}` | {st} | {r.get('returncode')} | {err} |")
    lines.append("")

    # Pipeline
    lines.append("## 3. Pipeline smoke (baseline, 500 rows)")
    lines.append("")
    pipeline_ok = True
    for name, args in PIPELINE_STEPS:
        lines.append(f"### `{name}`")
        lines.append("")
        r = run_subprocess(args, timeout=1800)
        if not r["ok"]:
            pipeline_ok = False
        lines.append(f"- **Status:** {'PASS' if r['ok'] else '**FAIL**'}")
        lines.append(f"- **returncode:** {r.get('returncode')}")
        if r.get("stderr_tail"):
            lines.append("")
            lines.append("<details><summary>stderr (tail)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(r["stderr_tail"][-3000:])
            lines.append("```")
            lines.append("</details>")
        lines.append("")

    # XLM-R optional (long)
    lines.append("## 4. XLM-R train smoke (1 epoch, batch 4)")
    lines.append("")
    lines.append("> May take several minutes and download ~1GB model on first run.")
    lines.append("")
    xlmr = run_subprocess(XLMR_SMOKE, timeout=7200)
    lines.append(f"- **Status:** {'PASS' if xlmr['ok'] else '**FAIL**'}")
    lines.append(f"- **returncode:** {xlmr.get('returncode')}")
    if xlmr.get("stderr_tail"):
        lines.append("")
        lines.append("<details><summary>stderr (tail)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(xlmr["stderr_tail"][-3000:])
        lines.append("```")
        lines.append("</details>")
    lines.append("")

    # Summary
    imp_fail = [r for r in import_rows if not r["ok"]]
    lines.append("## 5. Summary")
    lines.append("")
    lines.append(f"- Module import failures: **{len(imp_fail)}**")
    lines.append(f"- Baseline pipeline smoke: **{'PASS' if pipeline_ok else 'FAIL'}**")
    lines.append(f"- XLM-R smoke: **{'PASS' if xlmr['ok'] else 'FAIL'}**")
    lines.append("")
    lines.append("See also: [ERRORS_AND_FIXES.md](ERRORS_AND_FIXES.md)")

    out = ROOT / "docs" / "RUNTIME_CHECK_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    if imp_fail or not pipeline_ok or not xlmr["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
