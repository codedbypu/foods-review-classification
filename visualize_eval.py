"""
Build an interactive HTML report comparing models from evaluate.py JSON output.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import config

DEFAULT_INPUT = config.DEFAULT_EVAL_REPORT
DEFAULT_OUTPUT = config.DEFAULT_EVAL_VIZ

STAR_LABELS = ["1", "2", "3", "4", "5"]
MODEL_LABELS = {"baseline": "Baseline (TF-IDF + XGBoost)", "xlmr": "XLM-R"}

# Display order: higher-is-better vs lower-is-better
SUMMARY_METRICS = [
    ("mae", "MAE", "lower", "ดาว (ยิ่งต่ำยิ่งดี)"),
    ("rmse", "RMSE", "lower", "ดาว (ยิ่งต่ำยิ่งดี)"),
    ("accuracy", "Accuracy", "higher", "สัดส่วนทายถูก (ปัด 1–5)"),
    ("f1_macro", "F1 (macro)", "higher", "F1 เฉลี่ยทุกดาว"),
    ("f1_weighted", "F1 (weighted)", "higher", "F1 ถ่วงตามจำนวนตัวอย่าง"),
]

COLORS = {
    "baseline": "#5c6bc0",
    "xlmr": "#26a69a",
}


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _winner(a: float, b: float, direction: str) -> str:
    if direction == "lower":
        if a < b:
            return "baseline"
        if b < a:
            return "xlmr"
    else:
        if a > b:
            return "baseline"
        if b > a:
            return "xlmr"
    return "tie"


def build_summary_rows(report: dict) -> list[dict]:
    models = report["models"]
    if "baseline" not in models or "xlmr" not in models:
        raise ValueError("Report must contain both 'baseline' and 'xlmr' under 'models'.")

    rows = []
    b, x = models["baseline"], models["xlmr"]
    for key, label, direction, hint in SUMMARY_METRICS:
        bv, xv = b[key], x[key]
        w = _winner(bv, xv, direction)
        rows.append(
            {
                "key": key,
                "metric": label,
                "hint": hint,
                "baseline": bv,
                "xlmr": xv,
                "winner": w,
                "direction": direction,
            }
        )
    return rows


def build_f1_rows(report: dict) -> list[dict]:
    rows = []
    for star in STAR_LABELS:
        rows.append(
            {
                "star": star,
                "baseline": report["models"]["baseline"]["classification_report"][star][
                    "f1-score"
                ],
                "xlmr": report["models"]["xlmr"]["classification_report"][star]["f1-score"],
            }
        )
    return rows


def render_html(report: dict, source_path: str) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for visualization. Install with: pip install plotly"
        ) from exc

    summary = build_summary_rows(report)
    f1_rows = build_f1_rows(report)
    b_data = report["models"]["baseline"]
    x_data = report["models"]["xlmr"]
    n_samples = b_data["n_samples"]
    test_input = report.get("input", "—")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Chart 1: summary grouped bars ---
    metric_names = [r["metric"] for r in summary]
    fig_summary = go.Figure()
    fig_summary.add_trace(
        go.Bar(
            name=MODEL_LABELS["baseline"],
            x=metric_names,
            y=[r["baseline"] for r in summary],
            marker_color=COLORS["baseline"],
            text=[f"{r['baseline']:.3f}" for r in summary],
            textposition="outside",
        )
    )
    fig_summary.add_trace(
        go.Bar(
            name=MODEL_LABELS["xlmr"],
            x=metric_names,
            y=[r["xlmr"] for r in summary],
            marker_color=COLORS["xlmr"],
            text=[f"{r['xlmr']:.3f}" for r in summary],
            textposition="outside",
        )
    )
    fig_summary.update_layout(
        title="เปรียบเทียบ Metrics หลัก",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
        yaxis_title="ค่า metric",
    )

    # --- Chart 2: F1 per star ---
    fig_f1 = go.Figure()
    fig_f1.add_trace(
        go.Bar(
            name=MODEL_LABELS["baseline"],
            x=[f"ดาว {r['star']}" for r in f1_rows],
            y=[r["baseline"] for r in f1_rows],
            marker_color=COLORS["baseline"],
        )
    )
    fig_f1.add_trace(
        go.Bar(
            name=MODEL_LABELS["xlmr"],
            x=[f"ดาว {r['star']}" for r in f1_rows],
            y=[r["xlmr"] for r in f1_rows],
            marker_color=COLORS["xlmr"],
        )
    )
    fig_f1.update_layout(
        title="F1-score แยกตามดาว (1–5)",
        barmode="group",
        template="plotly_white",
        height=380,
        yaxis=dict(range=[0, 1], title="F1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
    )

    # --- Chart 3 & 4: confusion matrices ---
    def heatmap(cm: list, title: str) -> go.Figure:
        z = cm
        text = [[str(v) for v in row] for row in z]
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=[f"ทาย {s}" for s in STAR_LABELS],
                y=[f"จริง {s}" for s in STAR_LABELS],
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
            xaxis_title="ทาย (ปัดดาว)",
            yaxis_title="ดาวจริง",
            margin=dict(t=60, b=50),
        )
        return fig

    fig_cm_b = heatmap(b_data["confusion_matrix"], MODEL_LABELS["baseline"])
    fig_cm_x = heatmap(x_data["confusion_matrix"], MODEL_LABELS["xlmr"])

    summary_div = fig_summary.to_html(full_html=False, include_plotlyjs=False)
    f1_div = fig_f1.to_html(full_html=False, include_plotlyjs=False)
    cm_b_div = fig_cm_b.to_html(full_html=False, include_plotlyjs=False)
    cm_x_div = fig_cm_x.to_html(full_html=False, include_plotlyjs=False)

    # Comparison table rows
    table_rows = []
    for r in summary:
        w = r["winner"]
        b_cls = "winner" if w == "baseline" else ""
        x_cls = "winner" if w == "xlmr" else ""
        winner_label = {
            "baseline": "Baseline",
            "xlmr": "XLM-R",
            "tie": "เสมอ",
        }[w]
        badge_cls = w if w != "tie" else "tie"
        table_rows.append(
            f"""<tr>
  <td class="col-metric"><strong>{r['metric']}</strong><br><span class="hint">{r['hint']}</span></td>
  <td class="col-baseline num {b_cls}">{r['baseline']:.4f}</td>
  <td class="col-xlmr num {x_cls}">{r['xlmr']:.4f}</td>
  <td class="col-winner"><span class="badge {badge_cls}">{winner_label}</span></td>
</tr>"""
        )

    delta_acc = (x_data["accuracy"] - b_data["accuracy"]) * 100
    delta_mae = b_data["mae"] - x_data["mae"]

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Model Evaluation — Baseline vs XLM-R</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f4f6f9;
      --card: #ffffff;
      --text: #1a237e;
      --muted: #5c6b7a;
      --accent: #26a69a;
      --baseline: #5c6bc0;
      --xlmr: #26a69a;
      --winner-bg: #e8f5e9;
      --shadow: 0 4px 24px rgba(26, 35, 126, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(160deg, #e8eaf6 0%, var(--bg) 40%, #e0f2f1 100%);
      color: var(--text);
      margin: 0;
      padding: 2rem 1.5rem 3rem;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    header {{
      text-align: center;
      margin-bottom: 2rem;
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      letter-spacing: -0.02em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: var(--shadow);
      border-left: 4px solid var(--accent);
    }}
    .card.baseline {{ border-left-color: var(--baseline); }}
    .card.xlmr {{ border-left-color: var(--xlmr); }}
    .card h3 {{ margin: 0 0 0.5rem; font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card .value {{ font-size: 1.5rem; font-weight: 700; }}
    .card .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }}
    section {{
      background: var(--card);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow);
    }}
    section h2 {{
      font-size: 1.15rem;
      margin: 0 0 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid #e8eaf6;
    }}
    table.compare {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.95rem;
      border: 1px solid #cfd8dc;
      border-radius: 10px;
      overflow: hidden;
    }}
    table.compare th,
    table.compare td {{
      padding: 0.85rem 1.15rem;
      border-bottom: 1px solid #dce1e6;
      border-right: 1px solid #dce1e6;
      vertical-align: middle;
    }}
    table.compare th:last-child,
    table.compare td:last-child {{
      border-right: none;
    }}
    table.compare tbody tr:last-child td {{
      border-bottom: none;
    }}
    table.compare th {{
      color: var(--text);
      font-weight: 600;
      font-size: 0.9rem;
      letter-spacing: 0.02em;
    }}
    table.compare th.col-metric {{
      text-align: left;
      background: #eceff1;
      width: 32%;
    }}
    table.compare th.col-baseline {{
      text-align: right;
      background: #e8eaf6;
      border-top: 3px solid var(--baseline);
      width: 22%;
    }}
    table.compare th.col-xlmr {{
      text-align: right;
      background: #e0f2f1;
      border-top: 3px solid var(--xlmr);
      width: 22%;
    }}
    table.compare th.col-winner {{
      text-align: center;
      background: #f5f5f5;
      border-top: 3px solid #90a4ae;
      width: 14%;
    }}
    table.compare td.col-metric {{
      background: #fafafa;
      text-align: left;
    }}
    table.compare td.col-baseline {{
      background: #fafbff;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    table.compare td.col-xlmr {{
      background: #f7fcfb;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    table.compare td.col-winner {{
      background: #ffffff;
      text-align: center;
    }}
    table.compare td.col-baseline.winner {{
      background: #e8eaf6;
      font-weight: 700;
    }}
    table.compare td.col-xlmr.winner {{
      background: #e0f2f1;
      font-weight: 700;
    }}
    table.compare tbody tr:hover td.col-metric {{ background: #f0f0f0; }}
    table.compare tbody tr:hover td.col-baseline {{ background: #f5f6fc; }}
    table.compare tbody tr:hover td.col-xlmr {{ background: #f0faf8; }}
    table.compare tbody tr:hover td.col-winner {{ background: #fafafa; }}
    .badge {{
      display: inline-block;
      min-width: 4.5rem;
      padding: 0.25em 0.75em;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-align: center;
    }}
    .badge.baseline {{ background: #c5cae9; color: #1a237e; }}
    .badge.xlmr {{ background: #b2dfdb; color: #004d40; }}
    .badge.tie {{ background: #eceff1; color: #546e7a; }}
    .hint {{ font-size: 0.75rem; color: var(--muted); font-weight: normal; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }}
    @media (min-width: 900px) {{
      .charts-grid.two-col {{ grid-template-columns: 1fr 1fr; }}
    }}
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 2rem;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>เปรียบเทียบโมเดล — Baseline vs XLM-R</h1>
      <p class="meta">
        ชุดทดสอบ: <code>{test_input}</code> · n = {n_samples} ·
        สร้างจาก <code>{os.path.basename(source_path)}</code> · {generated}
      </p>
    </header>

    <div class="cards">
      <div class="card xlmr">
        <h3>Accuracy (XLM-R)</h3>
        <div class="value">{x_data['accuracy']*100:.1f}%</div>
        <div class="sub">สูงกว่า baseline +{delta_acc:.1f} pp</div>
      </div>
      <div class="card xlmr">
        <h3>MAE (XLM-R)</h3>
        <div class="value">{x_data['mae']:.3f}</div>
        <div class="sub">ต่ำกว่า baseline ประมาณ {delta_mae:.3f} ดาว</div>
      </div>
      <div class="card baseline">
        <h3>Baseline Accuracy</h3>
        <div class="value">{b_data['accuracy']*100:.1f}%</div>
        <div class="sub">TF-IDF + XGBoost</div>
      </div>
      <div class="card baseline">
        <h3>Baseline MAE</h3>
        <div class="value">{b_data['mae']:.3f}</div>
        <div class="sub">RMSE {b_data['rmse']:.3f}</div>
      </div>
    </div>

    <section>
      <h2>ตารางเปรียบเทียบ Metrics</h2>
      <table class="compare">
        <thead>
          <tr>
            <th class="col-metric">Metric</th>
            <th class="col-baseline">Baseline</th>
            <th class="col-xlmr">XLM-R</th>
            <th class="col-winner">ดีกว่า</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>กราฟ Metrics หลัก</h2>
      {summary_div}
    </section>

    <section>
      <h2>F1 แยกตามดาว</h2>
      {f1_div}
    </section>

    <section>
      <h2>Confusion Matrix (แถว = ดาวจริง, คอลัมน์ = ดาวที่ทาย)</h2>
      <div class="charts-grid two-col">
        <div>{cm_b_div}</div>
        <div>{cm_x_div}</div>
      </div>
    </section>

    <footer>
      สร้างโดย visualize_eval.py · เปิดไฟล์นี้ในเบราว์เซอร์
    </footer>
  </div>
</body>
</html>
"""
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize eval_report.json as an interactive HTML dashboard.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"JSON from evaluate.py (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}\nRun evaluate.py first.")

    report = load_report(args.input)
    html = render_html(report, args.input)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved visualization: {args.output}")
    print("Open this file in your browser to view charts and comparison table.")


if __name__ == "__main__":
    main()
