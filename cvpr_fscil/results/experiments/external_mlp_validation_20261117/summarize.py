from pathlib import Path
import json
import numpy as np

folder = Path(__file__).resolve().parent
summary = {}
lines = [
    "# Two-layer nonlinear MLP external-head validation",
    "",
    "Standalone experiment. Three benchmarks (CIFAR-100, miniImageNet, CUB-200), seeds 0/1/2, five calibration images per base class, retention ratio 0.95, joint error budget 0.05. Table-S8 splits, base-only alignment tuning and paired betting certification are reused. External head is Linear(input, 256) -> ReLU -> Linear(256, classes). Architecture fixed before evaluation. Class-weighted cross entropy, Adam learning rate 0.001, 300 full-batch steps, initialization seed 2026. Weight decay independently selected from [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] using base tuning accuracy, both for the deployed head and pseudo-base tuning head. Calibration and test data are excluded from fitting and selection. Both variants retain the same external probe and use the same 3,636-setting base-only search. Novel support is five-shot. Final-session results only.",
    "",
    "Per-query alignment and fixed-moment alignment each select their coefficients independently using base-only tuning data. All base-calibration recovery rates are asserted to be zero. Failed certifications contribute zero novel accuracy.",
    "",
    "Confidence intervals use 20,000 paired bootstrap resamples of the nine dataset-seed differences, with RNG seed 2026, matching the existing experiment. These intervals describe these nine paired units, with only three datasets and shared frozen features.",
    "",
    "| Encoder | Method | Admitted | Certified novel (%) | Conditional novel (%) | Paired improvement [95% CI] (pp) | Wins |",
    "|---|---|---|---|---|---|---|",
]
for enc in ["b16"]:
    payload = json.loads((folder / f"{enc}_records.json").read_text())
    records = payload["records"]["5"]
    ours = records["AnchorCE_full"]
    control = records["Wrap_fixed_mom"]
    keys = [(r["ds"], r["seed"]) for r in ours]
    assert len(keys) == 9 and len(set(keys)) == 9
    assert keys == [(r["ds"], r["seed"]) for r in control]
    differences = np.array([a["novel"] - b["novel"] for a, b in zip(ours, control)])
    rng = np.random.default_rng(2026)
    boot = differences[rng.integers(0, 9, size=(20000, 9))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    results = dict(improvement_pp=float(differences.mean()), ci95=[float(low), float(high)], wins=int((differences > 0).sum()), methods={})
    for name, data in records.items():
        admitted = [r["novel"] for r in data if r["admitted"]]
        result = dict(admitted=len(admitted), total=9, certified_novel=float(np.mean([r["novel"] for r in data])), conditional_novel=float(np.mean(admitted)) if admitted else None,
                      per_dataset={ds: float(np.mean([r["novel"] for r in data if r["ds"] == ds])) for ds in ["cifar100", "mini", "cub200"]})
        results["methods"][name] = result
        contrast = f'{differences.mean():.2f} [{low:.2f}, {high:.2f}]' if name == "AnchorCE_full" else "Reference"
        wins = str(results["wins"]) + "/9" if name == "AnchorCE_full" else ""
        lines.append(f'| {enc} | {name} | {len(admitted)}/9 | {result["certified_novel"]:.2f} | {result["conditional_novel"]:.2f} | {contrast} | {wins} |')
    results["reference_base_mean"] = float(np.mean([r["reference_base"] for r in ours]))
    for name, data in records.items():
        results["methods"][name]["empirical_floor_met"] = sum(r["empirical_floor_met"] for r in data)
        assert all(r["recovery_percent"] == 0 for r in data)
    summary[enc] = results
    lines.append("")
    lines.append(f'Mean deployed MLP base accuracy: {results["reference_base_mean"]:.2f}%.')
    for name, data in records.items():
        lines.append(f'{name} per-dataset certified novel accuracy: ' + str(results["methods"][name]["per_dataset"]))
        lines.append(f'{name} empirical test floors met: {results["methods"][name]["empirical_floor_met"]}/9.')

lines.extend(["", "The experiment tests the external-probe wrapper with four available sources. It does not independently isolate the necessity of each source. This is one nonlinear architecture with a prespecified width and optimizer schedule.", "", "Run commands", "", "```powershell", "python run_mlp_validation.py --encoder b16", "python summarize.py", "```"])
(folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[8:]))
