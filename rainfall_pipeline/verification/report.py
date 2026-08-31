"""Stage 6 -- the verification report.

Runs every model (A through E) over the same held-out test rows and emits a
single JSON file plus rendered Markdown and HTML summary tables.

The five entries, per the project brief:

===== =========================================================================
A     Raw NWP forecast, uncorrected.
B     One global ML bias-correction model, regime-blind.
C     Global empirical quantile mapping, no ML.
D     Regime-specific residual correction, no probability head.
E     Regime-specific residual correction + calibrated heavy-rain probability.
===== =========================================================================

D and E produce the *same* rainfall field -- the probability head does not
change the corrected amount -- so their continuous and rainfall-categorical
scores are identical by construction. E is distinguished by the probabilistic
scores (Brier score, reliability) that only it can produce, and by the fact that
its warning levels come from calibrated probabilities rather than from
thresholding a deterministic amount. The report states this explicitly rather
than letting a reader mistake two identical rows for a reproduced result.

No number in this module is hardcoded, defaulted or estimated. Every value comes
from running the metrics over real held-out data. If a model has not been
trained, its row is omitted and the omission is recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.thresholds import RAIN_THRESHOLDS, VERIFICATION
from ..data import schema as sch
from . import metrics as mx

#: Canonical model identifiers and their human-readable descriptions.
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "A_raw_nwp": "Baseline A -- raw NWP forecast, uncorrected",
    "B_global_ml": "Baseline B -- single global ML bias correction (regime-blind)",
    "C_quantile_mapping": "Baseline C -- global empirical quantile mapping (no ML)",
    "C_regime_quantile_mapping": "Ablation -- quantile mapping fitted per regime",
    "D_regime_residual": "Model D -- regime-specific residual correction",
    "E_regime_residual_probability": (
        "Model E -- regime-specific residual correction + calibrated heavy-rain probability"
    ),
}

#: Column the report groups by when the test set spans several forecast leads.
LEAD_TIME_COLUMN = "lead_time_steps"

#: Order the rows appear in the rendered tables.
MODEL_ORDER: List[str] = [
    "A_raw_nwp",
    "B_global_ml",
    "C_quantile_mapping",
    "C_regime_quantile_mapping",
    "D_regime_residual",
    "E_regime_residual_probability",
]


@dataclass
class VerificationInputs:
    """Everything the report needs about one evaluation run.

    Attributes:
        test_df: The held-out test rows, containing ``observed_mm``,
            ``date``/``lat``/``lon``, and optionally ``regime``, ``district``
            and ``region``.
        predictions: ``{model_id: corrected rainfall Series}``, each aligned to
            ``test_df.index``.
        probabilities: ``{model_id: {threshold_name: probability Series}}`` for
            models with a probability head. Optional.
        split_summary: Output of
            :meth:`~rainfall_pipeline.verification.splits.Split.summary`.
        notes: Free-form caveats to surface in the report.
    """

    test_df: pd.DataFrame
    predictions: Dict[str, pd.Series]
    probabilities: Dict[str, Dict[str, pd.Series]] = field(default_factory=dict)
    split_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def brier_score(observed_binary: pd.Series, probability: pd.Series) -> float:
    """Mean squared error of a probability forecast.

    Args:
        observed_binary: 1 if the event occurred, 0 otherwise.
        probability: Forecast probability in ``[0, 1]``.

    Returns:
        The Brier score (lower is better), or NaN if no finite pairs exist.
    """
    obs = pd.to_numeric(observed_binary, errors="coerce").to_numpy(dtype=float)
    prob = pd.to_numeric(probability, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(prob)
    if not mask.any():
        return float("nan")
    return float(np.mean((prob[mask] - obs[mask]) ** 2))


def brier_skill_score(observed_binary: pd.Series, probability: pd.Series) -> float:
    """Brier score relative to a climatological (constant base-rate) forecast.

    Args:
        observed_binary: 1 if the event occurred, 0 otherwise.
        probability: Forecast probability.

    Returns:
        BSS (1 is perfect, 0 is no better than climatology), or NaN if the
        reference is degenerate.
    """
    obs = pd.to_numeric(observed_binary, errors="coerce").dropna()
    if obs.empty:
        return float("nan")
    base = float(obs.mean())
    reference = base * (1.0 - base)
    if reference == 0:
        return float("nan")
    return float(1.0 - brier_score(observed_binary, probability) / reference)


def reliability_curve(
    observed_binary: pd.Series, probability: pd.Series, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Bin forecast probabilities and report the observed frequency in each bin.

    A well-calibrated model has ``observed_frequency ~= mean_probability`` in
    every bin. This is the table a reader should look at before trusting any
    probability the system emits.

    Args:
        observed_binary: 1 if the event occurred, 0 otherwise.
        probability: Forecast probability.
        n_bins: Number of equal-width probability bins.

    Returns:
        One dict per non-empty bin with ``bin_lower``, ``bin_upper``, ``n``,
        ``mean_probability`` and ``observed_frequency``.
    """
    obs = pd.to_numeric(observed_binary, errors="coerce")
    prob = pd.to_numeric(probability, errors="coerce")
    mask = obs.notna() & prob.notna()
    obs, prob = obs[mask], prob[mask]
    if obs.empty:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob.to_numpy(), edges[1:-1], right=False), 0, n_bins - 1)
    out: List[Dict[str, float]] = []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        out.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": float(sel.sum()),
                "mean_probability": float(prob.to_numpy()[sel].mean()),
                "observed_frequency": float(obs.to_numpy()[sel].mean()),
            }
        )
    return out


def evaluate_probabilities(
    test_df: pd.DataFrame, probabilities: Dict[str, pd.Series]
) -> Dict[str, Dict[str, Any]]:
    """Score a probability head against the observations.

    Args:
        test_df: Test rows containing ``observed_mm``.
        probabilities: ``{threshold_name: probability Series}``.

    Returns:
        ``{threshold_name: {"brier_score", "brier_skill_score", "base_rate",
        "reliability": [...]}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    obs = pd.to_numeric(test_df[sch.OBSERVED_COLUMN], errors="coerce")
    for name, prob in probabilities.items():
        mm = RAIN_THRESHOLDS.get(name)
        if mm is None:
            continue
        binary = (obs > mm).where(obs.notna()).astype("float64")
        out[name] = {
            "threshold_mm": float(mm),
            "base_rate": float(binary.mean(skipna=True)),
            "brier_score": brier_score(binary, prob),
            "brier_skill_score": brier_skill_score(binary, prob),
            "reliability": reliability_curve(binary, prob),
        }
    return out


def build_verification_report(
    inputs: VerificationInputs,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    stratify_by: Optional[List[str]] = None,
    max_districts: int = 50,
) -> Dict[str, Any]:
    """Compute every metric for every model and assemble the report dict.

    Args:
        inputs: The evaluation bundle.
        thresholds: ``{name: mm}``. Defaults to the IMD categories.
        stratify_by: Columns to break results down by. Defaults to
            ``["regime", "region", "intensity_bucket"]`` plus ``district`` when
            present.
        max_districts: Cap on the number of districts reported individually, to
            keep the JSON a sane size. The districts with the most test rows are
            kept.

    Returns:
        A JSON-serialisable report dict.

    Raises:
        ValueError: If no predictions were supplied, or ``observed_mm`` is
            missing from the test frame.
    """
    thresholds = dict(thresholds or RAIN_THRESHOLDS)
    if not inputs.predictions:
        raise ValueError("No model predictions supplied; nothing to verify.")
    if sch.OBSERVED_COLUMN not in inputs.test_df.columns:
        raise ValueError(
            f"Test frame has no '{sch.OBSERVED_COLUMN}' column, so nothing can "
            f"be verified against."
        )

    df = inputs.test_df.copy()
    df["intensity_bucket"] = mx.bucket_by_intensity(df, sch.OBSERVED_COLUMN)

    # Lead time is continuous in the schema but is only ever meaningful in
    # whole forecast steps, so bucket it before grouping. A dataset with a
    # single lead time produces a one-row table that says nothing, so it is
    # only stratified on when the forecasts actually span several leads.
    lead_values = (
        pd.to_numeric(df["lead_time"], errors="coerce")
        if "lead_time" in df.columns
        else pd.Series(dtype="float64")
    )
    has_lead_spread = lead_values.notna().any() and lead_values.nunique(dropna=True) > 1
    if has_lead_spread:
        df[LEAD_TIME_COLUMN] = lead_values.round().astype("Int64").astype("object")

    if stratify_by is None:
        stratify_by = [c for c in ("regime", "region", "intensity_bucket") if c in df.columns]
        if has_lead_spread:
            stratify_by.append(LEAD_TIME_COLUMN)
        if sch.DISTRICT_COLUMN in df.columns:
            stratify_by.append(sch.DISTRICT_COLUMN)

    # Keep only the busiest districts so the JSON stays readable.
    if sch.DISTRICT_COLUMN in stratify_by:
        top = (
            df[sch.DISTRICT_COLUMN]
            .value_counts()
            .head(max_districts)
            .index.astype(str)
            .tolist()
        )
    else:
        top = []

    models: Dict[str, Any] = {}
    for model_id, pred in inputs.predictions.items():
        col = f"__pred_{model_id}"
        df[col] = pd.to_numeric(pd.Series(np.asarray(pred), index=df.index), errors="coerce")

        entry: Dict[str, Any] = {
            "description": MODEL_DESCRIPTIONS.get(model_id, model_id),
            "overall": mx.evaluate(
                df, sch.OBSERVED_COLUMN, col, thresholds=thresholds, compute_fss=True
            ),
            "by": {},
        }
        for group_col in stratify_by:
            subset = df
            if group_col == sch.DISTRICT_COLUMN and top:
                subset = df[df[sch.DISTRICT_COLUMN].astype(str).isin(top)]
            entry["by"][group_col] = mx.evaluate_by_group(
                subset, sch.OBSERVED_COLUMN, col, group_col, thresholds=thresholds
            )

        probs = inputs.probabilities.get(model_id)
        if probs:
            entry["probabilistic"] = evaluate_probabilities(df, probs)
        models[model_id] = entry

    missing = [m for m in MODEL_ORDER if m not in models and m in MODEL_DESCRIPTIONS]
    notes = list(inputs.notes)
    if missing:
        notes.append(
            "Not evaluated (no trained artifact available at report time): "
            + ", ".join(missing)
        )
    if {"D_regime_residual", "E_regime_residual_probability"} <= set(models):
        notes.append(
            "Models D and E share the same corrected rainfall field, so their "
            "continuous and rainfall-categorical scores are identical by "
            "construction. E is distinguished only by the probabilistic scores."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_test_rows": int(len(df)),
        "test_period": {
            "start": str(df["date"].min().date()) if "date" in df.columns and not df.empty else None,
            "end": str(df["date"].max().date()) if "date" in df.columns and not df.empty else None,
        },
        "split": inputs.split_summary,
        "thresholds_mm": thresholds,
        "fss_neighborhood_sizes": list(VERIFICATION.fss_neighborhood_sizes),
        "stratified_by": stratify_by,
        "models": models,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table cell, rendering NaN/None as an em dash."""
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "--"
    return f"{f:.{digits}f}"


def summary_table(report: Dict[str, Any]) -> pd.DataFrame:
    """Build the headline model-comparison table.

    Args:
        report: Output of :func:`build_verification_report`.

    Returns:
        One row per model with RMSE, bias, correlation and the categorical
        scores at the ``heavy`` threshold.
    """
    rows = []
    for model_id in MODEL_ORDER:
        entry = report["models"].get(model_id)
        if entry is None:
            continue
        cont = entry["overall"]["continuous"]
        heavy = entry["overall"]["categorical"].get("heavy", {})
        fss_heavy = entry["overall"].get("fss", {}).get("heavy", {})
        row = {
            "model": model_id,
            "description": entry["description"],
            "n": cont.get("n"),
            "rmse_mm": cont.get("rmse"),
            "bias_mm": cont.get("bias"),
            "correlation": cont.get("correlation"),
            "pod_heavy": heavy.get("pod"),
            "far_heavy": heavy.get("far"),
            "csi_heavy": heavy.get("csi"),
            "ets_heavy": heavy.get("ets"),
        }
        for window, score in fss_heavy.items():
            row[f"fss_heavy_{window}"] = score
        if "probabilistic" in entry and "heavy" in entry["probabilistic"]:
            row["brier_heavy"] = entry["probabilistic"]["heavy"]["brier_score"]
            row["bss_heavy"] = entry["probabilistic"]["heavy"]["brier_skill_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as Markdown, ready to paste into a slide deck.

    Args:
        report: Output of :func:`build_verification_report`.

    Returns:
        The Markdown document as a string.
    """
    lines: List[str] = [
        "# Verification report",
        "",
        f"- Generated (UTC): {report['generated_at_utc']}",
        f"- Test rows: {report['n_test_rows']:,}",
        f"- Test period: {report['test_period']['start']} to {report['test_period']['end']}",
        f"- FSS neighbourhood sizes (grid cells): {report['fss_neighborhood_sizes']}",
        "",
    ]

    if report.get("split"):
        lines += ["## Chronological split", ""]
        lines += ["| Split | Rows | Start | End |", "|---|---:|---|---|"]
        for name, info in report["split"].items():
            lines.append(
                f"| {name} | {info.get('n_rows', 0):,} | {info.get('start')} | {info.get('end')} |"
            )
        lines.append("")

    table = summary_table(report)
    lines += ["## Model comparison (overall, `heavy` = 64.5 mm)", ""]
    if table.empty:
        lines += ["_No models were evaluated._", ""]
    else:
        headers = [c for c in table.columns if c != "description"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for _, row in table.iterrows():
            cells = [
                str(row[h]) if h == "model" else _fmt(row[h], 0 if h == "n" else 3)
                for h in headers
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Model key:", ""]
        for _, row in table.iterrows():
            lines.append(f"- `{row['model']}` -- {row['description']}")
        lines.append("")

    # Per-threshold categorical detail.
    lines += ["## Categorical skill by threshold", ""]
    lines += ["| Model | Threshold | POD | FAR | CSI | ETS | Freq. bias | Hits | Misses | False alarms |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id in MODEL_ORDER:
        entry = report["models"].get(model_id)
        if entry is None:
            continue
        for name, scores in entry["overall"]["categorical"].items():
            lines.append(
                "| {m} | {t} ({mm} mm) | {pod} | {far} | {csi} | {ets} | {fb} | {h} | {ms} | {fa} |".format(
                    m=model_id,
                    t=name,
                    mm=_fmt(report["thresholds_mm"].get(name), 1),
                    pod=_fmt(scores.get("pod")),
                    far=_fmt(scores.get("far")),
                    csi=_fmt(scores.get("csi")),
                    ets=_fmt(scores.get("ets")),
                    fb=_fmt(scores.get("frequency_bias")),
                    h=_fmt(scores.get("hits"), 0),
                    ms=_fmt(scores.get("misses"), 0),
                    fa=_fmt(scores.get("false_alarms"), 0),
                )
            )
    lines.append("")

    # Stratified breakdowns.
    for group_col in report.get("stratified_by", []):
        lines += [f"## RMSE by {group_col}", ""]
        group_values: List[str] = []
        for entry in report["models"].values():
            group_values += list(entry["by"].get(group_col, {}))
        group_values = sorted(set(group_values))
        if not group_values:
            lines += ["_No data._", ""]
            continue
        lines.append("| Model | " + " | ".join(group_values) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(group_values)) + "|")
        for model_id in MODEL_ORDER:
            entry = report["models"].get(model_id)
            if entry is None:
                continue
            cells = []
            for value in group_values:
                block = entry["by"].get(group_col, {}).get(value)
                cells.append(_fmt(block["continuous"]["rmse"], 2) if block else "--")
            lines.append(f"| {model_id} | " + " | ".join(cells) + " |")
        lines.append("")

    # Reliability of the probability head.
    for model_id, entry in report["models"].items():
        prob_block = entry.get("probabilistic")
        if not prob_block:
            continue
        lines += [f"## Probability calibration -- {model_id}", ""]
        lines += ["| Threshold | Base rate | Brier score | Brier skill score |", "|---|---:|---:|---:|"]
        for name, scores in prob_block.items():
            lines.append(
                f"| {name} | {_fmt(scores['base_rate'], 4)} | "
                f"{_fmt(scores['brier_score'], 4)} | {_fmt(scores['brier_skill_score'])} |"
            )
        lines.append("")
        for name, scores in prob_block.items():
            if not scores["reliability"]:
                continue
            lines += [f"### Reliability -- {name}", ""]
            lines += ["| Forecast prob. bin | n | Mean forecast | Observed frequency |", "|---|---:|---:|---:|"]
            for row in scores["reliability"]:
                lines.append(
                    f"| {row['bin_lower']:.1f}-{row['bin_upper']:.1f} | {int(row['n']):,} | "
                    f"{_fmt(row['mean_probability'])} | {_fmt(row['observed_frequency'])} |"
                )
            lines.append("")

    if report.get("notes"):
        lines += ["## Notes", ""]
        lines += [f"- {n}" for n in report["notes"]]
        lines.append("")
    return "\n".join(lines)


def render_html(report: Dict[str, Any]) -> str:
    """Render the report as a standalone HTML document.

    Args:
        report: Output of :func:`build_verification_report`.

    Returns:
        A complete HTML page as a string.
    """
    table = summary_table(report)
    body = (
        table.to_html(index=False, float_format=lambda v: f"{v:.3f}", na_rep="--")
        if not table.empty
        else "<p><em>No models were evaluated.</em></p>"
    )
    notes = "".join(f"<li>{n}</li>" for n in report.get("notes", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rainfall verification report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; line-height: 1.5; }}
  table {{ border-collapse: collapse; margin: 1rem 0; font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid #d0d0d0; padding: 0.4rem 0.7rem; text-align: right; }}
  th {{ background: #f4f4f4; text-align: left; }}
  td:first-child, th:first-child {{ text-align: left; }}
  code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>Verification report</h1>
<p>
  Generated (UTC): {report['generated_at_utc']}<br>
  Test rows: {report['n_test_rows']:,}<br>
  Test period: {report['test_period']['start']} to {report['test_period']['end']}
</p>
<h2>Model comparison (overall, heavy = 64.5 mm)</h2>
{body}
<h2>Notes</h2>
<ul>{notes}</ul>
<p><em>Full stratified results, per-threshold contingency tables and reliability
curves are in the accompanying JSON and Markdown files.</em></p>
</body>
</html>
"""


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        # JSON has no NaN/Infinity; null is the honest representation of an
        # undefined metric.
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def write_report(
    report: Dict[str, Any],
    *,
    json_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
    html_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write the report to disk in all three formats.

    Args:
        report: Output of :func:`build_verification_report`.
        json_path: Destination for the JSON. Defaults to the configured path.
        markdown_path: Destination for the Markdown. Defaults to the config.
        html_path: Destination for the HTML. Defaults to the config.

    Returns:
        ``{"json": path, "markdown": path, "html": path}``.
    """
    from ..config.regions import (
        VERIFICATION_HTML_PATH,
        VERIFICATION_MARKDOWN_PATH,
        VERIFICATION_REPORT_PATH,
        ensure_dirs,
    )

    ensure_dirs()
    json_path = Path(json_path or VERIFICATION_REPORT_PATH)
    markdown_path = Path(markdown_path or VERIFICATION_MARKDOWN_PATH)
    html_path = Path(html_path or VERIFICATION_HTML_PATH)

    json_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path, "html": html_path}


def load_report(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load a previously written verification report.

    Args:
        path: The JSON file. Defaults to the configured path.

    Returns:
        The report dict, or None if no report has been generated yet.
    """
    from ..config.regions import VERIFICATION_REPORT_PATH

    path = Path(path or VERIFICATION_REPORT_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
