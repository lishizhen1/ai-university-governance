"""Read-only checks of published aggregates; no clinical data or fitted models.

Run from any directory with Python 3.12 or later. This checks arithmetic and
consistency, not patient-level reproduction or model refitting.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def near(a, b, tolerance=1e-8):
    assert abs(float(a) - float(b)) < tolerance, (a, b)


def quantile(values, q):
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def main():
    point = rows(HERE / "ici_point_estimates.csv")
    table = rows(HERE / "ici_e50_e90_with_95CI.csv")
    bootstrap = rows(HERE / "ici_bootstrap_metric_replicates.csv")
    pairs = rows(HERE / "ici_e50_e90_paired_differences.csv")
    assert (len(point), len(table), len(bootstrap), len(pairs)) == (26, 26, 13000, 72)
    grouped = {}
    estimates = {(r["cohort"], r["model"]): r for r in point}
    for row in bootstrap:
        grouped.setdefault((row["cohort"], row["model"]), []).append(row)
    assert len(grouped) == 26
    metrics = ["ICI", "E50", "E90"]
    for key, sample in grouped.items():
        assert len(sample) == 500
        assert {int(r["bootstrap"]) for r in sample} == set(range(500))
        sample.sort(key=lambda r: int(r["bootstrap"]))
        assert all(0 <= float(r[m]) <= 1 for r in sample for m in metrics)
        assert all(float(r["E50"]) <= float(r["E90"]) for r in sample)
    for row in table:
        key = row["cohort"], row["model"]
        assert int(row["bootstrap_replicates"]) == 500
        assert int(row["n"]) == (10810 if key[0] == "MIMIC-IV" else 4355)
        for metric in metrics:
            near(row[metric], estimates[key][metric])
            near(row[metric + "_lo"], quantile([r[metric] for r in grouped[key]], .025))
            near(row[metric + "_hi"], quantile([r[metric] for r in grouped[key]], .975))
    for row in pairs:
        a = row["cohort"], row["model_a"]
        b = row["cohort"], row["model_b"]
        metric = row["metric"]
        difference = [float(x[metric]) - float(y[metric]) for x, y in zip(grouped[a], grouped[b])]
        near(row["difference"], float(estimates[a][metric]) - float(estimates[b][metric]))
        near(row["lo"], quantile(difference, .025))
        near(row["hi"], quantile(difference, .975))

    report = json.loads((HERE / "latent_class" / "FINAL_LATENT_REPORT.json").read_text(encoding="utf-8"))
    latent_results = []
    for variant, folder, n in [
        ("observed", HERE / "latent_class", 10810),
        ("complete", HERE / "latent_class" / "complete_case", 6826),
    ]:
        comparison = rows(folder / "model_comparison.csv")
        initializations = rows(folder / "model_initialization_diagnostics.csv")
        quality = rows(folder / "classification_quality.csv")
        classes = rows(folder / "class_summary.csv")
        assert (len(comparison), len(initializations), len(quality), len(classes)) == (4, 139, 8, 20)
        assert {int(r["k"]) for r in comparison} == {1, 2, 3, 4}
        for row in comparison:
            k = int(row["k"])
            p = int(row["parameters"])
            likelihood = float(row["loglik"])
            assert int(row["n_subjects"]) == n and int(row["conv"]) == 1
            near(row["AIC"], -2 * likelihood + 2 * p)
            near(row["BIC"], -2 * likelihood + p * math.log(n))
            assert all(float(row[x]) < 1e-5 for x in ["gconv_parameters", "gconv_likelihood", "gconv_derivatives"])
            full = [r for r in initializations if int(r["k"]) == k and r["phase"] == "full" and int(r["conv"]) == 1]
            assert len(full) == (1 if k == 1 else 6)
            count = sum(abs(float(r["loglik"]) - likelihood) <= 1e-4 for r in full)
            assert count == int(row["best_loglik_repeats_tolerance_1e_4"])
            reference = next(r for r in report[variant]["model_comparison"] if r["k"] == k)
            for metric in ["AIC", "BIC", "loglik", "n_subjects", "conv"]:
                near(row[metric], reference[metric])
        best = int(min(comparison, key=lambda r: float(r["BIC"]))["k"])
        assert best == report[variant]["best_BIC_k_among_valid_tested"]
        for row in quality:
            group = [r for r in classes if r["k"] == row["k"] and r["dataset"] == row["dataset"]]
            total = int(row["n"])
            assert sum(int(r["assigned_n"]) for r in group) == total
            near(sum(float(r["posterior_expected_n"]) for r in group), total)
            assert min(int(r["assigned_n"]) for r in group) == int(row["min_assigned_class_n"])
            near(sum(int(r["assigned_n"]) * float(r["average_posterior_assigned"]) for r in group) / total, row["mean_max_posterior"])
            for threshold in ["0_7", "0_8"]:
                weighted = sum(int(r["assigned_n"]) * float(r["pct_assigned_posterior_ge_" + threshold]) / 100 for r in group) / total
                near(weighted, 1 - float(row["proportion_max_posterior_lt_" + threshold]))
            reference = next(r for r in report[variant]["classification_quality"] if r["k"] == int(row["k"]) and r["dataset"] == row["dataset"])
            for key, value in row.items():
                if key != "dataset" and value != "NA":
                    near(value, reference[key])
        latent_results.append({"variant": variant, "models": 4, "lowest_BIC_within_K1_K4": best})

    manifest = json.loads((HERE.parent / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = HERE.parent / entry["path"]
        content = path.read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], entry["path"]
    print(json.dumps({
        "status": "PASS",
        "scope": "Published aggregate arithmetic and file hashes only; no clinical data, models or refits loaded.",
        "calibration_model_cohort_estimates": 26,
        "calibration_bootstrap_rows": 13000,
        "paired_metric_contrasts": 72,
        "latent_analyses": latent_results,
        "public_manifest_entries_verified": len(manifest["files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
