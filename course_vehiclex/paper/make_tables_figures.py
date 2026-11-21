"""Build every table and figure of the course paper from the saved VehicleX run JSONs.

Usage (from vehiclex_course/):  python paper/make_tables_figures.py
Inputs : anchorce_fscil_baselines_metrics.json  (AnchorCE, HyCal, TEEN; 101 offset candidates)
         anchorce_dense_metrics.json            (GDA, GDA-Anchor; 101 offset candidates)
         anchorce_matched_metrics.json          (AnchorCE, GDA, GDA-Anchor; denser candidates for all)
         metrics.json                           (dataset inventory)
Outputs: paper/tab_*.tex, paper/fig_*.pdf, paper/numbers.json
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CAPS = ("5", "10", "40")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


base101 = load("anchorce_fscil_baselines_metrics.json")
gda101 = load("anchorce_dense_metrics.json")
dense = load("anchorce_matched_metrics.json")
inventory = load("metrics.json")["inventory"]["attributes"]["type"]

# Method name in the paper -> (run file, key inside that file)
STD = {"TEEN": (base101, "TEEN"), "HyCal": (base101, "HyCal"), "GDA": (gda101, "GDA"),
       "GDA-Anchor": (gda101, "GDA-Anchor"), "AnchorCE": (base101, "AnchorCE")}
DENSE = {"GDA": (dense, "GDA"), "GDA-Anchor": (dense, "GDA-Anchor"), "AnchorCE": (dense, "AnchorCE")}
ORDER = list(STD)
COLORS = {"TEEN": "#8c6d31", "HyCal": "#7b4173", "GDA": "#d6616b", "GDA-Anchor": "#e6a141", "AnchorCE": "#3182bd"}

# Sanity check: the AnchorCE rows of the two 101-candidate files are the same runs.
for r1, r2 in zip(base101["runs"], gda101["runs"]):
    assert r1["seed"] == r2["seed"]
    a = r1["sessions"][-1]["methods"]["AnchorCE"]["unconstrained"]["overall"]
    b = r2["sessions"][-1]["methods"]["AnchorCE"]["unconstrained"]["overall"]
    assert abs(a - b) < 1e-12
SEEDS = [r["seed"] for r in base101["runs"]]


def per_seed(src, key, fn):
    return np.array([fn(run, run["sessions"][-1]["methods"][key]) for run in src["runs"]])


def ms(x, digits=1):
    x = 100 * np.asarray(x, float)
    return f"{x.mean():.{digits}f}$\\pm${x.std(ddof=1):.{digits}f}"


def session_curve(src, key, part):
    curves = []
    for run in src["runs"]:
        row = []
        for s in run["sessions"]:
            m = s["methods"][key]
            if "closed_base_macro" in m:
                row.append(m["closed_base_macro"] if part in ("overall", "base") else np.nan)
            else:
                row.append(m["unconstrained"][part])
        curves.append(row)
    return 100 * np.array(curves)


numbers = {}

# ---------------------------------------------------------------- Table: dataset
names = ["sedan", "SUV", "van", "hatchback", "MPV", "pickup", "bus", "truck", "wagon", "sports car", "RV"]
r0 = base101["runs"][0]
base_ids = r0["base"]
rows = []
for k in sorted(range(11), key=lambda k: -inventory["vehicles"][str(k)]):
    role = "base" if k in base_ids else "novel"
    rows.append(f"{names[k]} & {role} & {inventory['vehicles'][str(k)]} & {inventory['images'][str(k)]:,} \\\\")
(OUT / "tab_dataset.tex").write_text(
    "\\begin{tabular}{llrr}\\toprule\nType & Role & Vehicles & Images \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")
roles = {k: len(v) for k, v in r0["role_vehicle_ids"].items()}
numbers["roles_seed42_vehicles"] = roles
numbers["roles_seed42_images"] = {k: len(v) for k, v in r0["role_indices"].items()}

# ---------------------------------------------------------------- Table: unconstrained accuracy
lines = []
for name in ORDER:
    src, key = STD[name]
    closed = np.array([run["sessions"][0]["methods"][key]["closed_base_macro"] for run in src["runs"]])
    ov = session_curve(src, key, "overall") / 100
    avg, last = ov.mean(1), ov[:, -1]
    fb = per_seed(src, key, lambda r, m: m["unconstrained"]["base"])
    fn = per_seed(src, key, lambda r, m: m["unconstrained"]["novel"])
    lines.append(f"{name} & {ms(closed)} & {ms(avg)} & {ms(last)} & {ms(fb)} & {ms(fn)} \\\\")
    numbers[f"unc_{name}"] = dict(closed=100 * closed.mean(), avg=100 * avg.mean(), last=100 * last.mean(),
                                  base=100 * fb.mean(), novel=100 * fn.mean(), last_by_seed=(100 * last).round(2).tolist())
(OUT / "tab_unconstrained.tex").write_text(
    "\\begin{tabular}{lccccc}\\toprule\n"
    "Method & $A^{\\mathrm{closed}}_{B}$ & Avg & Last & Last base & Last novel \\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")


# ---------------------------------------------------------------- Table: certification (101 candidates)
def cert(src, key, contract, cap):
    adm = per_seed(src, key, lambda r, m: m["contracts"][contract]["budgets"][cap]["admitted"]).astype(int)
    nov = per_seed(src, key, lambda r, m: m["contracts"][contract]["budgets"][cap]["novel"])
    met = per_seed(src, key, lambda r, m: m["contracts"][contract]["budgets"][cap]["meets_test_target"]).astype(int)
    return adm, nov, met


def oracle(src, key, contract):
    return per_seed(src, key, lambda r, m: m["contracts"][contract]["oracle"].get("novel", 0.0)
                    if m["contracts"][contract]["oracle"]["feasible"] else 0.0)


def recovery(src, key, contract, cap="40"):
    g = per_seed(src, key, lambda r, m: m["contracts"][contract]["budgets"][cap]["closed_winner_recovery"])
    d = per_seed(src, key, lambda r, m: m["contracts"][contract]["budgets"][cap]["first_test"]["destroyed"])
    return g, d


lines = []
all_low_budget_zero = True
for name in ORDER:
    src, key = STD[name]
    cells = [name]
    for contract in ("self", "common"):
        for cap in ("5", "10"):
            adm, _, _ = cert(src, key, contract, cap)
            all_low_budget_zero &= adm.sum() == 0
        g, d = recovery(src, key, contract)
        adm, nov, met = cert(src, key, contract, "40")
        cells += [f"{100 * g.mean():.1f}", f"{100 * d.mean():.1f}", f"{100 * oracle(src, key, contract).mean():.1f}",
                  f"{adm.sum()}/6", f"{100 * nov.mean():.1f}"]
        numbers[f"cert101_{name}_{contract}"] = dict(G=100 * g.mean(), D=100 * d.mean(),
                                                    oracle=100 * oracle(src, key, contract).mean(),
                                                    admitted=int(adm.sum()), novel=100 * nov.mean(), met=int(met.sum()),
                                                    novel_by_seed=(100 * nov).round(2).tolist())
    lines.append(" & ".join(cells) + " \\\\")
numbers["all_methods_zero_at_5_and_10"] = bool(all_low_budget_zero)
(OUT / "tab_certification.tex").write_text(
    "\\begin{tabular}{l@{\\hskip 4pt}ccccc@{\\hskip 6pt}ccccc}\\toprule\n"
    "& \\multicolumn{5}{c}{Self reference (own head)} & \\multicolumn{5}{c}{Common reference (AnchorCE head)} \\\\"
    "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
    "Method & $G$ & $D_1$ & Oracle & Adm. & Cert. & $G$ & $D_1$ & Oracle & Adm. & Cert. \\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- Table: candidate density ablation
lines = []
for name in ("GDA", "GDA-Anchor", "AnchorCE"):
    cells = [name]
    for contract in ("self", "common"):
        a101, n101, _ = cert(*STD[name], contract, "40")
        ad, nd, metd = cert(*DENSE[name], contract, "40")
        for cap in ("5", "10"):
            assert cert(*DENSE[name], contract, cap)[0].sum() == 0
        cells += [f"{a101.sum()}/6 \\,/\\, {100 * n101.mean():.1f}", f"{ad.sum()}/6 \\,/\\, {100 * nd.mean():.1f}",
                  f"{100 * (nd - n101).mean():+.1f}"]
        numbers[f"dense_{name}_{contract}"] = dict(adm101=int(a101.sum()), cert101=100 * n101.mean(),
                                                  adm_dense=int(ad.sum()), cert_dense=100 * nd.mean(),
                                                  gain=100 * (nd - n101).mean(), met=int(metd.sum()),
                                                  dense_by_seed=(100 * nd).round(2).tolist())
    lines.append(" & ".join(cells) + " \\\\")
(OUT / "tab_dense.tex").write_text(
    "\\begin{tabular}{lcccccc}\\toprule\n"
    "& \\multicolumn{3}{c}{Self reference} & \\multicolumn{3}{c}{Common reference} \\\\"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
    "Method & 101 & Dense & $\\Delta$ & 101 & Dense & $\\Delta$ \\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- Minimum calibration budget at zero recovery
r, delta, T = base101["ratio"], base101["delta"], 5
LAMBDAS = np.array([.01, .025, .05, .1, .2, .3, .4, .5, .65, .8, .95, 1.0])  # grid of the certification code


def best_log_e(k):
    """Largest mixture e-value when k correct reference predictions are all kept and no error is repaired."""
    return math.log(np.mean((1 + LAMBDAS * (1 - r)) ** k))


k_min = next(k for k in range(1, 2000) if best_log_e(k) >= math.log(T / delta))
closed_anchor = numbers["unc_AnchorCE"]["closed"] / 100
numbers["min_budget"] = dict(k_min=k_min, threshold=math.log(T / delta),
                             per_class_at_anchor_acc=k_min / (6 * closed_anchor),
                             best_log_e_5=best_log_e(round(30 * closed_anchor)),
                             best_log_e_10=best_log_e(round(60 * closed_anchor)),
                             best_log_e_40=best_log_e(round(240 * closed_anchor)))

# ---------------------------------------------------------------- Figure: session curves
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9))
for ax, part, title in zip(axes, ("overall", "base", "novel"), ("All classes", "Base classes", "New classes")):
    for name in ORDER:
        c = session_curve(*STD[name], part)
        keep = ~np.isnan(c).all(0)
        c, x = c[:, keep], np.arange(c.shape[1])[keep]
        mean, sd = c.mean(0), c.std(0, ddof=1)
        ax.plot(x, mean, marker="o", ms=2.5, lw=1.2, color=COLORS[name], label=name)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[name], alpha=.15, lw=0)
    ax.set_title(title)
    ax.set_xlabel("Session")
    ax.set_xticks(range(6))
axes[0].set_ylabel("Macro accuracy (%)")
axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=.3)
fig.savefig(OUT / "fig_sessions.pdf")

# ---------------------------------------------------------------- Figure: certification overview
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), sharey=True)
for ax, contract, title in zip(axes, ("self", "common"), ("Self reference", "Common reference")):
    names_ = ORDER
    x = np.arange(len(names_))
    v101 = [100 * cert(*STD[n], contract, "40")[1].mean() for n in names_]
    vden = [100 * cert(*DENSE[n], contract, "40")[1].mean() if n in DENSE else np.nan for n in names_]
    ax.bar(x - .2, v101, .4, color=[COLORS[n] for n in names_], label="101 candidates")
    ax.bar(x + .2, vden, .4, color=[COLORS[n] for n in names_], alpha=.45, hatch="///", label="dense")
    ax.set_xticks(x)
    ax.set_xticklabels(["TEEN", "HyCal", "GDA", "GDA-A", "Anch."], rotation=40, fontsize=6.5)
    ax.set_title(title)
axes[0].set_ylabel("Certified new acc. (%)")
axes[1].legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(pad=.3)
fig.savefig(OUT / "fig_certification.pdf")

(OUT / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
print(json.dumps(numbers, indent=1))

# ---------------------------------------------------------------- Table for the CVPR-based course version
# Deployed-head contract: every method must retain the Session-0 deployed head (AnchorCE's closed head),
# 40 calibration images per base type, 101 candidate offsets, final session, mean over six seeds.
rows = []
for name in ORDER:
    u = numbers[f"unc_{name}"]
    c = numbers[f"cert101_{name}_common"]
    cells = [f"{u['avg']:.1f}", f"{u['last']:.1f}", f"{c['G']:.1f}", f"{c['D']:.1f}", f"{c['admitted']}/6", f"{c['novel']:.1f}"]
    label = "\\textbf{\\textsc{AnchorCE}}" if name == "AnchorCE" else name
    if name == "AnchorCE":
        cells = [f"\\textbf{{{x}}}" for x in cells]
    rows.append(label + " & " + " & ".join(cells) + " \\\\")
(ROOT / "paper_cvpr").mkdir(exist_ok=True)
(ROOT / "paper_cvpr" / "table_vehiclex_deployed.tex").write_text(
    "\\setlength{\\tabcolsep}{0pt}\n\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}\n\\toprule\n"
    "Method & Avg & Last & $G$ & $D_1$ & Adm. & Cert. \\\\\n\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\n\\end{tabular*}\n", encoding="utf-8")

# ---------------------------------------------------------------- Supplement tables for the CVPR-based course version
CV = ROOT / "paper_cvpr"
CV.mkdir(exist_ok=True)
(CV / "table_vehiclex_dataset.tex").write_text((OUT / "tab_dataset.tex").read_text(encoding="utf-8"), encoding="utf-8")
import shutil
# Supplement figure: all-class accuracy per session (the headline unconstrained metric).
fig, ax = plt.subplots(figsize=(3.3, 2.0))
for name in ORDER:
    c = session_curve(*STD[name], "overall")
    mean, sd = c.mean(0), c.std(0, ddof=1)
    x = np.arange(c.shape[1])
    ax.plot(x, mean, marker="o", ms=2.5, lw=1.4 if name == "AnchorCE" else 1.0, color=COLORS[name], label=name)
    ax.fill_between(x, mean - sd, mean + sd, color=COLORS[name], alpha=.12, lw=0)
ax.set_xlabel("Session")
ax.set_ylabel("Accuracy on all seen types (%)")
ax.set_xticks(range(6))
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=.3)
fig.savefig(CV / "fig_vehiclex_sessions.pdf")

# Per-seed AnchorCE upgrades of the deployed head (40 calibration images per base type).
rows = []
for r101, rden in zip(base101["runs"], dense["runs"]):
    m = r101["sessions"][-1]["methods"]["AnchorCE"]
    c101 = m["contracts"]["common"]["budgets"]["40"]
    cden = rden["sessions"][-1]["methods"]["AnchorCE"]["contracts"]["common"]["budgets"]["40"]
    ref = m["contracts"]["common"]["reference_base"]
    rows.append(f"{r101['seed']} & {100 * ref:.1f} & {100 * m['unconstrained']['overall']:.1f} & "
                f"{100 * c101['first_test']['recovered']:.1f} & {100 * cden['base']:.1f} & "
                f"{100 * cden['novel']:.1f}{'' if cden['admitted'] else '$^\\dagger$'} & "
                f"{'\\checkmark' if cden['meets_test_target'] else '--'} \\\\")
(CV / "table_vehiclex_perseed.tex").write_text(
    "\\begin{tabular}{lcccccc}\\toprule\n"
    "Seed & $A_{\\mathrm{ref}}$ & Last & $G$ & $A_B(\\hat d)$ & Cert. & Met \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")

# Budget floor: attainable vs. observed first-candidate log e-value of AnchorCE under the deployed head.
rows = []
for cap, n in (("5", 30), ("10", 60), ("40", 240)):
    obs = per_seed(base101, "AnchorCE", lambda r, m: m["contracts"]["common"]["budgets"][cap]["first_test"]["log_evalue"])
    k = round(n * closed_anchor)
    rows.append(f"{cap} & {n} & {k} & {best_log_e(k):.2f} & {obs.mean():.2f} & {math.log(T / delta):.2f} \\\\")
(CV / "table_vehiclex_budget.tex").write_text(
    "\\begin{tabular}{cccccc}\\toprule\n"
    "Img./type & $n$ & $k$ & Max $\\log E$ & Obs. $\\log E$ & $\\log(T/\\delta)$ \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n", encoding="utf-8")
