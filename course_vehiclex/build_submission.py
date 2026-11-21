"""Build the course code-and-results package (AnchorCE_code.zip).

Contents
  course_vehiclex/          VehicleX experiments of this course project (Table 6, Supplement G)
  cvpr_fscil/code/          code of the CEC / held-out-domain experiments (Tables 1-5, Supplement A-F)
  cvpr_fscil/results/       per-run and summary results read by the table/figure scripts

Excluded on purpose: CLIP feature caches (*.pt), dataset archives (*.zip, *.tgz),
model weights (*.safetensors), score caches (*.npz) that no table script reads,
and cluster-orchestration scripts (SSH/remote control), which are not experiment code.
"""
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../科研新
COURSE = ROOT / "vehiclex_course"
JLT = ROOT / "JLT-Head"
MINI = "--mini" in sys.argv          # course limit for supplementary material: 20 MB
LITE = "--lite" in sys.argv or MINI  # OpenReview single-file limit: 100 MB
OUT = COURSE / "submission" / ("AnchorCE_supplementary.zip" if LITE else "AnchorCE_code.zip")
# In the lite package, per-run outputs are replaced by the summary-level files of each study.
HEAVY = {"expansion_prospective_validation", "kbs_reviewer_revision", "kbs_full_fusion_retuning",
         "expansion_constant_sensitivity_remote", "expansion_shrinkage"}   # summary-only in the lite package
SUMMARY_NAME = re.compile(r"(summary|verification|report|results|findings|readme|protocol|constants|complete)", re.I)

# Result directories referenced by the paper's table and figure scripts.
RESULT_DIRS = [
    "autodl_round4_20261109", "autodl_round5_20261110", "merged_ref4_20261110", "autodl_anchorg_20261109",
    "autodl_extra_baselines_20261109", "autodl_descriptions_20261109",
    "kbs_independent_risk", "kbs_reviewer_revision", "kbs_mechanism_followup", "kbs_full_fusion_retuning",
    "kbs_strengthening", "expansion_cardinality_rule_v2", "expansion_cross_encoder", "expansion_cub_remote",
    "expansion_followup_diagnostics", "expansion_prospective_validation", "expansion_reference_cardinality",
    "expansion_rn50", "expansion_shrinkage", "expansion_softmax_readout", "expansion_constant_sensitivity_remote",
    "expansion_hycal_self_remote", "expansion_reference_completion", "expansion_self_reference_remote",
    "expansion_shared_head_remote", "expansion_sun_controls_remote", "expansion_gate_calibration", "fcs", "fcs_pets",
    "expansion_equal_budget_external_smoke", "expansion_independent_reference_replay",
    "expansion_prospective_external_smoke", "external_mlp_validation_20261117",
    "external_probe_encoder_validation_20261117", "kbs_imagenetr",
    "kbs_retention_safety_diagnostic", "kbs_sketch_acquisition", "kbs_sketch_validation", "local_recovery_20261110",
]
NPZ_DIRS = {"kbs_independent_risk", "expansion_gate_calibration"}   # *.npz read directly by table scripts
TEXT_EXT = {".json", ".md", ".txt", ".csv", ".log", ".tex"}
CODE_EXT = {".py", ".sh"}
SENSITIVE = re.compile(r"seetacloud|paramiko|password|key_filename|C:/Users/|\.ssh/|AUTODL_EXTRA", re.I)
SKIP_PARTS = {"__pycache__", "_build_linux", "cvpr2027_revisions", "before_cpu_overlap_20261108",
              "before_desc_execution_20261109", "before_fecam_replay2", "before_throughput_20261108",
              "before_true_fecam_only_20261108"}
COURSE_FILES = ["README.md", "run.py", "run_anchor_vehiclex.py", "build_submission.py", "metrics.json",
                "run.log", "samples.json",
                "anchorce_dense.log", "anchorce_dense_metrics.json", "anchorce_matched.log",
                "anchorce_matched_metrics.json", "anchorce_fscil_baselines.log",
                "anchorce_fscil_baselines_metrics.json", "paper/make_tables_figures.py", "paper/numbers.json"]

# Working notes (revision plans, hand-offs, drafts, run diaries) are not code or results.
INTERNAL = re.compile(r"(REVISION|HANDOFF|RESULTS_ZH|results_log|comparison_table|DRAFT|correction|_scope|"
                      r"CPU_GPU_PROFILE|FINDINGS|REPORT_ZH|PREFLIGHT|TWO_NODES)", re.I)
excluded_sensitive = []


def is_sensitive(path):
    try:
        return bool(SENSITIVE.search(path.read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return True


def add(zf, src, arc):
    if src.suffix.lower() == ".md" and INTERNAL.search(src.name):
        return 0
    # Class-description files and the CUB report match the filter by wording only; they hold no access details.
    allowed = src.name in {"build_submission.py", "mini.json", "modified_miniimagent_output_final.json"} or (
        src.name == "REPORT_ZH.md" and src.parent.name == "expansion_cub_remote")
    if not allowed and src.suffix.lower() in CODE_EXT | TEXT_EXT and is_sensitive(src):
        excluded_sensitive.append(str(src.relative_to(ROOT)))
        return 0
    zf.write(src, arc)
    return src.stat().st_size


def walk(base, exts, npz=False):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
        for name in filenames:
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext in exts or (npz and ext == ".npz"):
                yield p


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(COURSE / "submission" / "README_PACKAGE.md", "README.md")
        if LITE:
            zf.write(COURSE / "paper_cvpr" / "supplementary.pdf", "supplementary.pdf")
        for rel in COURSE_FILES:
            total += add(zf, COURSE / rel, f"course_vehiclex/{rel}")
        for src, arc in ((JLT / "autodl_fscil", "cvpr_fscil/code/autodl_fscil"),
                         (JLT / "src", "cvpr_fscil/code/src"),
                         (JLT / "paper" / "cvpr2027", "cvpr_fscil/code/paper_scripts")):
            for p in walk(src, CODE_EXT | ({".md"} if "autodl" in arc else set())):
                if src.name == "cvpr2027" and p.parent not in (src, src / "figures_src"):
                    continue
                total += add(zf, p, f"{arc}/{p.relative_to(src).as_posix()}")
            if src.name == "autodl_fscil":
                for p in walk(src / "descriptions", {".json", ".txt"}):
                    total += add(zf, p, f"{arc}/{p.relative_to(src).as_posix()}")
        top = JLT / "experiments"
        for p in sorted(top.iterdir()):           # protocols and top-level summaries of every study
            if p.is_file() and p.suffix in {".md", ".json"} and p.stat().st_size < 20e6:
                total += add(zf, p, f"cvpr_fscil/results/experiments/{p.name}")
        for d in RESULT_DIRS:
            base = JLT / "experiments" / d
            if not base.is_dir():
                print("missing", d, file=sys.stderr)
                continue
            for p in walk(base, TEXT_EXT | CODE_EXT, npz=d in NPZ_DIRS and not LITE):
                if LITE and (MINI or d in HEAVY) and p.suffix in TEXT_EXT and not (p.suffix == ".md" or SUMMARY_NAME.search(p.name)):
                    continue
                total += add(zf, p, f"cvpr_fscil/results/experiments/{p.relative_to(base.parent).as_posix()}")
    (COURSE / "submission" / "excluded_sensitive.txt").write_text("\n".join(excluded_sensitive), encoding="utf-8")
    print(f"raw {total / 1e6:.1f} MB -> zip {OUT.stat().st_size / 1e6:.1f} MB; "
          f"{len(excluded_sensitive)} files excluded for containing remote-access details")


if __name__ == "__main__":
    main()
