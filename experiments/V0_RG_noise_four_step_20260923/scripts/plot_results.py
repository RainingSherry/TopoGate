"""Render real discovery or frozen-probe figures; unavailable panels are skipped."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from summarize import write_csv, read_json, probe_paths

CASES = ("Mouse_retina", "dermatology", "har", "micro-mass")
RED, GRAY = "#B5465D", "#797A8E"
VARIANT_LABELS = {"mixed_observation_reconstruction": "Mixed-target reconstruction",
                  "no_edge_modulation": "No edge modulation",
                  "fixed_mix_original_aux": "Fixed mixing amplitude",
                  "base_scmae": "No neighbourhood branch", "full": "Full"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .7, "legend.frameon": False})


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        value = float(row[key])
        return value if np.isfinite(value) else None
    except (KeyError, ValueError, TypeError):
        return None


def save(fig, out, stem, source, caption, manifest):
    if manifest.get("smoke_validation_only"):
        fig.text(.5, -.045, "SMOKE VALIDATION ONLY · Limited coverage; not a mechanism conclusion",
                 ha="center", fontsize=7, color=GRAY)
        caption = "**Smoke validation only. This figure checks the pipeline, not a research hypothesis.**\n\n" + caption
    files = []
    for ext in ("svg", "pdf", "png"):
        target = out / f"{stem}.{ext}"
        fig.savefig(target, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.10)
        files.append(str(target.resolve()))
    write_csv(out / f"{stem}_source.csv", source)
    (out / f"{stem}_caption.md").write_text(caption, encoding="utf-8")
    plt.close(fig)
    manifest["generated"].append({"figure": stem, "files": files, "source_rows": len(source),
                                  "visual_qa": "Inspect PNG at final size; not automatically certified."})


def snapshot(path, out, manifest):
    rows = read_csv(path)
    chosen = [r for r in rows if r["dataset"] in CASES and r["corruption"] == "replacement"
              and np.isclose(float(r["severity"]), .2)]
    counts = {ds: sum(r["dataset"] == ds for r in chosen) for ds in CASES}
    if any(n != 1 for n in counts.values()):
        raise ValueError(f"Snapshot requires exactly one actual row per case: {counts}")
    chosen.sort(key=lambda r: CASES.index(r["dataset"]))
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 146 / 25.4))
    fig.subplots_adjust(left=.09, right=.99, bottom=.20, top=.83, hspace=.63, wspace=.33)
    source = []
    for col, row in enumerate(chosen):
        ds, upper, lower = row["dataset"], axes[0, col], axes[1, col]
        upper.set_title(ds.replace("_", " "))
        for label, suffix, color, marker, style in (("Micro", "micro", GRAY, "o", "-"),
                                                    ("Macro", "macro", RED, "s", "--")):
            vals = [float(row[f"{phase}_purity_{suffix}"]) for phase in ("clean", "noisy")]
            upper.plot([0, 1], vals, color=color, marker=marker, linestyle=style, ms=4, lw=1.3)
            source.extend(dict(dataset=ds, panel="purity", series=label, phase=phase, value=value)
                          for phase, value in zip(("clean", "noisy"), vals))
        upper.set_ylim(0, 1.04)
        upper.text(.5, -.28, f"kNN retention = {float(row['knn_retention']):.3f}",
                   transform=upper.transAxes, ha="center", fontsize=6.5)
        for label, ari, delta, color, marker in (("ToPoGate", "ari_tg", "delta_tg", RED, "o"),
                                                (row["best_baseline"], "ari_best", "delta_best", GRAY, "s")):
            vals = [float(row[ari]) - float(row[delta]), float(row[ari])]
            lower.plot([0, 1], vals, color=color, marker=marker, ms=4, lw=1.3,
                       linestyle="-" if label == "ToPoGate" else "--")
            source.extend(dict(dataset=ds, panel="ARI", series=label, phase=phase, value=value)
                          for phase, value in zip(("clean", "noisy"), vals))
        lower.set_ylim(0, 1.04)
        lower.text(.5, -.28, f"Comparator: {row['best_baseline']}", transform=lower.transAxes,
                   ha="center", fontsize=6.5)
        if col == 0:
            upper.set_ylabel("Neighbour purity")
            lower.set_ylabel("ARI")
        for ax in (upper, lower):
            ax.set_xlim(-.2, 1.2)
            ax.set_xticks([0, 1], ["Clean", "Noisy"])
            ax.grid(axis="y", color=".9", lw=.5)
    for ax, letter in zip(axes.flat, "abcdefgh"):
        ax.text(-.12, 1.05, letter, transform=ax.transAxes, weight="bold", fontsize=9)
    fig.suptitle("Selected advantage cases: neighbourhood change and clustering quality", y=.975, fontsize=10)
    fig.text(.5, .925, "Exploratory discovery · Replacement p = 0.2 · Frozen encoder and clean centroids",
             ha="center", fontsize=7)
    handles = [Line2D([], [], color=GRAY, marker="o", label="Micro purity"),
               Line2D([], [], color=RED, marker="s", ls="--", label="Macro purity"),
               Line2D([], [], color=RED, marker="o", label="ToPoGate ARI"),
               Line2D([], [], color=GRAY, marker="s", ls="--", label="Comparator ARI")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=4, fontsize=6.5)
    fig.text(.5, .035, "Legacy geometry: Euclidean model-input kNN, all rows; not the training graph.\n"
             "Purity uses seed 42; ARI is the reported aggregate. Selected cases do not establish a mechanism.",
             ha="center", fontsize=6.5)
    caption = ("# Exploratory discovery snapshot\n\nFour selected advantage cases under independent per-entry replacement p=0.2. "
               "Top: micro/macro neighbourhood purity from legacy seed-42 frozen-preprocessed Euclidean, all-row diagnostics; "
               "this is not the model PCA/cosine training graph. Bottom: reported ToPoGate and strongest noisy-result comparator ARI. "
               "Clean ARI = reported noisy ARI minus reported noisy-minus-clean delta; comparator selection used discovery outcomes. "
               "ARI aggregation and structural diagnostics have different replication scopes; these plotted points have no uncertainty bars. "
               "Lines connect clean/noisy conditions, not model-level replicates. HAR's near-tie is not a decisive advantage. "
               "These descriptive selected-case results do not establish causal mechanism or broad robustness. "
               "The invalid legacy raw changed-entry fraction is excluded.\n")
    save(fig, out, "discovery_preview", source, caption, manifest)


def metric_row(rows, metric, ds, corruption, severity, variant="full"):
    found = [r for r in rows if r.get("metric") == metric and r.get("dataset") == ds
             and r.get("variant") == variant and r.get("stage", "observe") == "observe"
             and r.get("corruption") == corruption and r.get("severity") == severity]
    return found[0] if len(found) == 1 else None


def mechanism(root, out, manifest):
    path = root / "summary.csv"
    if not path.exists():
        manifest["skipped"].append({"figure": "mechanism", "reason": "summary.csv missing"})
        return
    rows = read_csv(path)
    specs = [("input.model.normalized_displacement", "Input displacement / clean scale"),
             ("representation.normalized_drift", "Latent displacement / clean scale"),
             ("prediction.noisy_ari", "ARI")]
    # Never create an empty panel: only metrics present in every selected case are included.
    cases = [ds for ds in CASES if any(r.get("dataset") == ds and r.get("variant") == "full" and r.get("stage") == "observe" for r in rows)]
    valid_specs = [(m, label) for m, label in specs if cases and all(any(r.get("dataset") == ds and r.get("metric") == m
                        and r.get("variant") == "full" and r.get("stage") == "observe" and number(r, "mean") is not None for r in rows) for ds in cases)]
    for m, _ in specs:
        if m not in [s[0] for s in valid_specs]:
            manifest["skipped"].append({"figure": "mechanism:" + m, "reason": "missing valid metric in available cases"})
    if not cases or not valid_specs:
        return
    fig, axes = plt.subplots(len(valid_specs), len(cases), squeeze=False,
                             figsize=(183 / 25.4, (55 * len(valid_specs) + 20) / 25.4))
    fig.subplots_adjust(left=.12, right=.985, top=.88, bottom=.14, wspace=.5, hspace=.7)
    source = []
    for col, ds in enumerate(cases):
        conditions = sorted({(r["corruption"], r["severity"]) for r in rows if r.get("dataset") == ds
                             and r.get("variant") == "full" and r.get("stage") == "observe"})
        for ri, (metric, label) in enumerate(valid_specs):
            ax = axes[ri, col]
            for i, (kind, severity) in enumerate(conditions):
                row = metric_row(rows, metric, ds, kind, severity)
                if row is None or number(row, "mean") is None:
                    continue
                ax.errorbar(i, float(row["mean"]), yerr=number(row, "model_seed_sd"),
                            color=RED, marker="o", ms=3, capsize=2)
                source.append(row)
                if metric == "prediction.noisy_ari":
                    clean = metric_row(rows, "prediction.clean_ari", ds, kind, severity)
                    if clean and number(clean, "mean") is not None:
                        ax.scatter(i, float(clean["mean"]), color=GRAY, marker="_", s=60)
                        source.append(clean)
            ax.set_xticks(range(len(conditions)), [f"{'G' if k == 'gaussian' else 'R'} {s}" for k, s in conditions],
                          rotation=40, ha="right", fontsize=6)
            ax.grid(axis="y", color=".9", lw=.5)
            if ri == 0:
                ax.set_title(ds.replace("_", " "))
            if col == 0:
                ax.set_ylabel(label)
            ax.text(-.2, 1.06, chr(97 + ri * len(cases) + col), transform=ax.transAxes, weight="bold")
    fig.suptitle("Frozen-model responses in selected cases", fontsize=10)
    fig.text(.5, .035, "Red: full model; grey ticks: clean ARI. Bars: SD of model-seed means.\n"
             "Noise seeds averaged within each model seed; G/R severities are not matched damage.", ha="center", fontsize=6.5)
    save(fig, out, "mechanism_observation", source,
         "# Frozen-model observation\n\nInput and latent displacement are reported in their respective clean-normalized spaces. "
         "The full-dimensional displacement is not inferred from a 2-D layout. Red points are equally weighted model-seed means "
         "after averaging noise seeds. Error bars are sample SD across model-seed means, not confidence intervals; missing SD is omitted. "
         "Within-model noise SD and exact replication counts are in the source CSV. Grey ticks are clean ARI. "
         "Noise severities are protocol values, not a claim of equal damage. These selected cases and descriptive associations do not establish mediation.\n", manifest)


def input_structure(root, out, manifest):
    path = root / "summary.csv"
    rows = read_csv(path) if path.exists() else []
    specs = [("representation.graph.neighbor_retention", "Neighbour retention"),
             ("representation.graph.purity_micro_change", "Micro purity change"),
             ("representation.graph.purity_macro_change", "Macro purity change")]
    observations = []
    for ds in CASES:
        conditions = sorted({(r["corruption"], r["severity"]) for r in rows if r.get("dataset") == ds
                             and r.get("stage") == "observe" and r.get("variant") == "full"})
        for kind, severity in conditions:
            damage = metric_row(rows, "input.model.normalized_displacement", ds, kind, severity)
            if damage is None or number(damage, "mean") is None:
                continue
            for metric, _ in specs:
                target = metric_row(rows, metric, ds, kind, severity)
                if target is not None and number(target, "mean") is not None:
                    observations.append(dict(target, input_normalized_displacement=damage["mean"]))
    cases = [ds for ds in CASES if all(any(r["dataset"] == ds and r["metric"] == m for r in observations) for m, _ in specs)]
    if not cases:
        manifest["skipped"].append({"figure": "input_structure", "reason": "No complete real input/graph metric chain"})
        return
    fig, axes = plt.subplots(3, len(cases), squeeze=False, figsize=(183 / 25.4, 155 / 25.4))
    fig.subplots_adjust(left=.13, right=.98, top=.89, bottom=.16, hspace=.5, wspace=.55)
    source = []
    for col, ds in enumerate(cases):
        for ri, (metric, label) in enumerate(specs):
            ax = axes[ri, col]
            for r in observations:
                if r["dataset"] != ds or r["metric"] != metric:
                    continue
                marker = "o" if r["corruption"] == "gaussian" else "s"
                ax.scatter(float(r["input_normalized_displacement"]), float(r["mean"]),
                           color=RED if marker == "o" else GRAY, marker=marker, s=16)
                source.append(r)
            if col == 0:
                ax.set_ylabel(label)
            if ri == 0:
                ax.set_title(ds.replace("_", " "))
            if ri == 2:
                ax.set_xlabel("Normalized input damage")
            if ri > 0:
                ax.axhline(0, color=GRAY, lw=.6, ls=":")
            ax.grid(axis="y", color=".9", lw=.5)
    fig.suptitle("Input damage and neighbourhood structure", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=RED, marker="o", linestyle="none", label="Gaussian"),
                        Line2D([], [], color=GRAY, marker="s", linestyle="none", label="Replacement")],
               loc="lower center", bbox_to_anchor=(.5, .05), ncol=2, fontsize=7)
    fig.text(.5, .015, "Selected cases · Full checkpoints · Descriptive seed means, not matched-damage causal effects", ha="center", fontsize=6.5)
    save(fig, out, "input_structure", source,
         "# Input damage and neighbourhood structure\n\nEach point is one corruption condition, averaged within model seed then across model seeds. "
         "Horizontal position is RMS input displacement normalized by clean RMS scale. Vertical metrics describe the clean-fitted graph "
         "diagnostic reported by the probe, not the legacy all-row Euclidean snapshot. Graph replay scope and status remain in probe records. "
         "Purity is a post-hoc label-based diagnostic only. Gaussian circles and replacement squares are not claimed to be equally damaging. "
         "These are descriptive associations in selected cases; uncertainty and replication counts remain in summary tables.\n", manifest)


def ablations(root, out, manifest):
    path = root / "paired_summary.csv"
    rows = read_csv(path) if path.exists() else []
    rows = [r for r in rows if r.get("stage") == "observe" and number(r, "mean") is not None]
    if not rows:
        manifest["skipped"].append({"figure": "effective_ablations", "reason": "No strictly paired effective interventions"})
        return
    cases = [ds for ds in CASES if any(r["dataset"] == ds for r in rows)]
    fig, axes = plt.subplots(len(cases), 1, squeeze=False, figsize=(183 / 25.4, (48 * len(cases) + 20) / 25.4))
    fig.subplots_adjust(left=.48, right=.96, top=.86, bottom=.22, hspace=.65)
    for ax, ds in zip(axes.flat, cases):
        selected = [r for r in rows if r["dataset"] == ds]
        for i, r in enumerate(selected):
            ax.errorbar(float(r["mean"]), i, xerr=number(r, "model_seed_sd"), color=RED, marker="o", ms=3, capsize=2)
        ax.set_yticks(range(len(selected)), [f"{VARIANT_LABELS.get(r['variant'], r['variant'])} | {r['corruption']} {r['severity']}" for r in selected], fontsize=6)
        ax.axvline(0, color=GRAY, ls="--", lw=.8)
        ax.set_title(ds, loc="left")
        ax.invert_yaxis()
    axes[-1, 0].set_xlabel("Full ARI change − ablated ARI change")
    save(fig, out, "effective_ablations", rows,
         "# Effective paired interventions\n\nOnly inventory-audited effective controls with identical protocol, preprocessing, test identities, "
         "source data and corrupted input hashes enter. Points average noise realizations within model seed then average model seeds; "
         "bars are SD across model-seed means. Positive values indicate a smaller ARI decline for full. "
         "This is a configuration contrast, not evidence of two-factor synergy or causal mediation. Excluded/no-op pairs are in pairing_rejections.csv.\n", manifest)
    # Give every condition enough vertical space for an independently reviewable case figure.
    for ds in cases:
        selected = [r for r in rows if r["dataset"] == ds]
        height_mm = max(83, 26 + 8 * len(selected))
        separate, ax = plt.subplots(figsize=(183 / 25.4, height_mm / 25.4))
        separate.subplots_adjust(left=.49, right=.97, top=.92, bottom=.11)
        for i, row in enumerate(selected):
            ax.errorbar(float(row["mean"]), i, xerr=number(row, "model_seed_sd"),
                        color=RED, marker="o", ms=3.5, capsize=2)
        ax.set_yticks(range(len(selected)),
                      [f"{VARIANT_LABELS.get(row['variant'], row['variant'])} | "
                       f"{'G' if row['corruption'] == 'gaussian' else 'R'} {row['severity']}"
                       for row in selected], fontsize=7)
        ax.axvline(0, color=GRAY, ls="--", lw=.8)
        ax.set_title(ds.replace("_", " "))
        ax.set_xlabel("Full ARI change − ablated ARI change")
        ax.invert_yaxis()
        ax.grid(axis="x", color=".92", lw=.5)
        save(separate, out, "effective_ablations_" + ds, selected,
             "# Effective paired interventions in " + ds + "\n\nPositive values mean a smaller ARI decline for full than for the indicated "
             "inventory-effective control. Each point averages noise seeds within model seed, then equally weights model seeds; "
             "bars are model-seed SD, not confidence intervals. G/R nominal severities are not damage-matched; "
             "the complete paired clean/noisy endpoints and coverage are in paired_ablations.csv and paired_summary.csv. "
             "Configuration contrasts do not establish mediation or two-factor synergy.\n", manifest)


def directions(root, out, manifest):
    path = root / "summary.csv"
    rows = read_csv(path) if path.exists() else []
    metrics = [("representation.common_valid.normalized_drift", "Normalized latent drift"),
               ("prediction.common_valid.flip_rate", "Assignment flip fraction")]
    paired_path = root / "paired_summary.csv"
    pairing = read_csv(paired_path) if paired_path.exists() else []
    pair_keys = ("protocol", "dataset", "variant", "corruption", "severity")
    effective = {tuple(r.get(k) for k in pair_keys) for r in pairing if r.get("stage") == "directions"}
    rows = [r for r in rows if r.get("stage") == "directions" and number(r, "mean") is not None
            and (r.get("variant") == "full" or tuple(r.get(k) for k in pair_keys) in effective)]
    cases = [ds for ds in CASES if all(any(r["dataset"] == ds and r["variant"] == "full" and r["metric"] == metric for r in rows)
                                     for metric, _ in metrics)]
    if not cases:
        manifest["skipped"].append({"figure": "matched_directions", "reason": "No real common-valid direction metrics"})
        return
    variants = ["full"] + sorted({r["variant"] for r in rows if r["variant"] != "full"})
    markers = dict(zip(variants, ("o", "s", "^", "D", "v", "P", "X")))
    fig, axes = plt.subplots(2, len(cases), squeeze=False, figsize=(183 / 25.4, 140 / 25.4))
    fig.subplots_adjust(left=.12, right=.98, bottom=.30, top=.88, wspace=.5, hspace=.48)
    source = []
    for col, ds in enumerate(cases):
        for ri, (metric, ylabel) in enumerate(metrics):
            ax = axes[ri, col]
            for name, color, style in (("native", RED, "-"), ("random", GRAY, "--"), ("shuffled", GRAY, ":")):
                for variant in variants:
                    series = sorted([r for r in rows if r["dataset"] == ds and r["metric"] == metric
                                     and r["variant"] == variant and r["corruption"] == "direction_" + name],
                                    key=lambda r: float(r["severity"]))
                    if not series:
                        continue
                    x, y = [float(r["severity"]) for r in series], [float(r["mean"]) for r in series]
                    ax.plot(x, y, color=color, ls=style, marker=markers[variant], ms=3.5,
                            markerfacecolor=color if variant == "full" else "white", lw=1 if variant == "full" else .65)
                    for xi, yi, r in zip(x, y, series):
                        if number(r, "model_seed_sd") is not None:
                            ax.errorbar(xi, yi, yerr=float(r["model_seed_sd"]), color=color, capsize=2)
                        counts = [n for n in rows if n["metric"] == "prediction.common_valid.n"
                                  and all(n.get(k) == r.get(k) for k in pair_keys)]
                        source.append(dict(r, mean_common_valid_n=counts[0]["mean"] if len(counts) == 1 else "unavailable"))
            if col == 0:
                ax.set_ylabel(ylabel)
            if ri == 0:
                ax.set_title(ds.replace("_", " "))
            else:
                ax.set_xlabel("Matched-norm multiplier")
            ax.grid(axis="y", color=".9", lw=.5)
    handles = [Line2D([], [], color=c, ls=ls, label=n) for n, c, ls in
               (("Native", RED, "-"), ("Random", GRAY, "--"), ("Shuffled", GRAY, ":"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .17), ncol=3, fontsize=6.5)
    variant_handles = [Line2D([], [], color=GRAY, linestyle="none", marker=markers[v],
                              markerfacecolor=GRAY if v == "full" else "white", label=VARIANT_LABELS.get(v, v)) for v in variants]
    fig.legend(handles=variant_handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, fontsize=6)
    fig.suptitle("Response to clean-derived, matched-norm directions", fontsize=10)
    fig.text(.5, .02, "Actual float32 common-valid rows · Frozen models · Effective controls only\n"
             "Source CSV reports mean common-valid n; bars are model-seed SD.", ha="center", fontsize=6.5)
    save(fig, out, "matched_directions", source,
         "# Matched-norm directions\n\nNative, random and shuffled directions use shared full-model-derived input coordinates and "
         "matched per-row norms. The common mask is computed after actual float32 perturbation: all three families must have nonzero "
         "realized displacement and satisfy the recorded norm tolerance. Metrics use this intersection, not nominal direction validity. "
         "Filled circles denote full; hollow distinct markers denote inventory-effective, strictly paired ablations. "
         "The source CSV reports mean common-valid sample count across model seeds. "
         "These model-input interventions differ from raw-input corruption experiments. Points average model-seed responses; bars show "
         "model-seed SD, not independent noise replicates or confidence intervals. A smaller native-direction response is evidence "
         "consistent with directional tolerance, not proof of training-path mediation or all-direction robustness.\n", manifest)


def layouts(root, out, manifest, limit):
    meta_path = root / "summary_manifest.json"
    summary_meta = read_json(meta_path) if meta_path.exists() else {}
    if "accepted_results" not in summary_meta:
        manifest["skipped"].append({"figure": "paired_layout", "reason": "Validated accepted_results manifest unavailable; rerun summarize.py on existing probes"})
        return
    data_root = Path(summary_meta["results_root"])
    accepted = {str(Path(p).resolve()) for p in summary_meta["accepted_results"]}
    made, seen = 0, set()
    for result_path in probe_paths(data_root):
        if made >= limit:
            break
        if str(result_path.resolve()) not in accepted:
            continue
        r = read_json(result_path)
        npz = result_path.with_name("samples.npz")
        if (r.get("dataset") in seen or r.get("dataset") not in CASES
                or r.get("status") not in ("ok", "success", "complete", "completed") or r.get("variant") != "full"
                or r.get("stage", "observe") != "observe" or r.get("corruption") != "replacement"
                or not np.isclose(float(r.get("severity", -1)), .2) or not npz.exists()):
            continue
        if r.get("representation", {}).get("projection_status") != "clean_fit_shared_axes":
            manifest["skipped"].append({"figure": "layout", "source": str(npz), "reason": "clean-fitted shared projection not verified"})
            continue
        with np.load(npz, allow_pickle=False) as sample:
            needed = ("clean_projection_2d", "noisy_projection_2d", "test_indices")
            if not all(k in sample for k in needed):
                manifest["skipped"].append({"figure": "layout", "source": str(npz), "reason": "paired clean-fitted projections unavailable"})
                continue
            clean, noisy, ids = (sample[k] for k in needed)
            if clean.shape != noisy.shape or clean.ndim != 2 or clean.shape[1] != 2 or len(ids) != len(clean):
                raise ValueError(f"Invalid paired layout shape: {npz}")
            if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
                continue
            take = np.linspace(0, len(clean) - 1, min(150, len(clean)), dtype=int)
            c, n = clean[take], noisy[take]
            fig, ax = plt.subplots(figsize=(89 / 25.4, 82 / 25.4))
            fig.subplots_adjust(left=.18, bottom=.19, top=.85, right=.97)
            ax.scatter(c[:, 0], c[:, 1], s=7, color=GRAY, alpha=.65, label="Clean")
            ax.scatter(n[:, 0], n[:, 1], s=7, color=RED, alpha=.65, label="Noisy")
            ax.quiver(c[:, 0], c[:, 1], n[:, 0] - c[:, 0], n[:, 1] - c[:, 1], angles="xy", scale_units="xy",
                      scale=1, color=GRAY, alpha=.35, width=.003)
            ax.set(xlabel="Clean-fitted PC1", ylabel="Clean-fitted PC2", title=f"{r['dataset']} · replacement 0.2")
            ax.legend(fontsize=6)
            source = [dict(test_index=int(ids[j]), clean_pc1=float(clean[j, 0]), clean_pc2=float(clean[j, 1]),
                           noisy_pc1=float(noisy[j, 0]), noisy_pc2=float(noisy[j, 1])) for j in take]
            stem = f"layout_{r['dataset']}_m{r['model_seed']}_n{r['noise_seed']}"
            save(fig, out, stem, source,
                 "# Paired displacement illustration\n\nUses probe-provided coordinates from PCA fitted once on clean embeddings; "
                 "the same transform projects noisy embeddings. A deterministic index-spaced subset of at most 150 test rows is shown, "
                 "without label-based selection or colouring. Arrows represent matched identities. This 2-D illustration is not evidence "
                 "of full-dimensional information preservation; it is one model/noise realization, not an aggregate.\n", manifest)
            made += 1
            seen.add(r["dataset"])
    if not made:
        manifest["skipped"].append({"figure": "paired_layout", "reason": "No eligible real paired projections"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--results", type=Path, help="Directory produced by summarize.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-layouts", type=int, default=4)
    p.add_argument("--smoke-validation", action="store_true", help="Visibly label all pipeline-validation figures")
    args = p.parse_args()
    if not args.snapshot and not args.results:
        p.error("Provide --snapshot and/or --results; no synthetic preview is generated.")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"backend": "Python matplotlib", "generated": [], "skipped": [],
                "smoke_validation_only": args.smoke_validation,
                "contract": "Selected-case exploratory/mechanism diagnostics; no fabricated data or inference of broad robustness."}
    if args.snapshot:
        snapshot(args.snapshot, args.out, manifest)
    if args.results:
        mechanism(args.results, args.out, manifest)
        input_structure(args.results, args.out, manifest)
        ablations(args.results, args.out, manifest)
        directions(args.results, args.out, manifest)
        layouts(args.results, args.out, manifest, args.max_layouts)
    else:
        manifest["skipped"].append({"figure": "mechanism/ablation/layout", "reason": "Probe results not supplied; discovery preview only"})
    (args.out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
