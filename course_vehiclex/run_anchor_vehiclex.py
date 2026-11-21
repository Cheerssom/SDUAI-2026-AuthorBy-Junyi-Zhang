"""Run the four-source AnchorCE readout on VehicleX's cached CLIP embeddings."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch


TYPE_NAMES = [
    "sedan", "sport utility vehicle", "van", "hatchback", "multi-purpose vehicle",
    "pickup truck", "bus", "truck", "station wagon", "sports car", "recreational vehicle",
]
TEMPLATES = [
    "a photo of a {}.", "a bad photo of a {}.", "a cropped photo of a {}.",
    "a close-up photo of a {}.", "a bright photo of a {}.", "a good photo of a {}.",
    "a photo of the {}.", "a photo of a {}.",
]
QUANTILES = np.linspace(0, 1, 101)
CALIBRATION_CAPS = (5, 10, 40)
METHODS = ("AnchorCE", "GDA", "GDA-Anchor")
FSCIL_METHODS = ("AnchorCE", "HyCal", "TEEN")
ADMISSION_VARIANTS = ("AnchorCE-BaseMargin", "AnchorCE-Bet1", "AnchorCE-DenseOffsets")


def protocol(samples, seed):
    labels = np.array([int(row["label"]) for row in samples])
    vehicles = np.array([row["vehicle"] for row in samples])
    grouped = defaultdict(list)
    for i, vehicle in enumerate(vehicles):
        grouped[vehicle].append(i)
    by_class = defaultdict(list)
    for vehicle, indices in sorted(grouped.items()):
        class_ids = np.unique(labels[indices])
        if len(class_ids) != 1:
            raise ValueError(f"Inconsistent type for vehicle {vehicle}")
        by_class[int(class_ids[0])].append(vehicle)
    if set(by_class) != set(range(len(TYPE_NAMES))):
        raise ValueError("The archive does not cover all 11 VehicleX type labels")
    base = np.array(sorted(sorted(by_class, key=lambda k: (-len(by_class[k]), k))[:6]))
    rng = np.random.default_rng(seed)
    arrival = rng.permutation(sorted(set(by_class) - set(base)))
    splits = {name: [] for name in ("fit", "tune", "cal", "support", "test")}
    fit_ids_by_class = {}
    for k, ids in sorted(by_class.items()):
        ids = rng.permutation(ids)
        n_test = max(1, round(.20 * len(ids)))
        test_ids, train_ids = ids[:n_test], ids[n_test:]
        splits["test"].extend(i for key in test_ids for i in grouped[key])
        if k in base:
            tune, cal, fit = train_ids[:8], train_ids[8:48], train_ids[48:]
            if len(tune) != 8 or len(cal) != 40 or len(fit) < 5:
                raise ValueError(f"Insufficient vehicle IDs in base type {k}")
            fit_ids_by_class[k] = fit.tolist()
            splits["tune"].extend(i for key in tune for i in rng.choice(grouped[key], 4, replace=False))
            splits["cal"].extend(int(rng.choice(grouped[key])) for key in cal)
            splits["fit"].extend(i for key in fit for i in grouped[key])
        else:
            if len(train_ids) < 5:
                raise ValueError(f"Insufficient support vehicle IDs in novel type {k}")
            splits["support"].extend(int(rng.choice(grouped[key])) for key in train_ids[:5])
    splits = {name: np.array(indices, dtype=int) for name, indices in splits.items()}
    vehicle_sets = {name: set(vehicles[ix]) for name, ix in splits.items()}
    all_role_ids = set().union(*vehicle_sets.values())
    assert len(all_role_ids) == sum(len(ids) for ids in vehicle_sets.values())
    pseudo_base = np.sort(rng.permutation(base)[:len(base) // 2])
    pseudo_novel = np.array(sorted(set(base) - set(pseudo_base)))
    pseudo_support = np.array([
        int(rng.choice(grouped[key]))
        for k in pseudo_novel
        for key in rng.choice(fit_ids_by_class[k], 5, replace=False)
    ])
    return labels, base, arrival, pseudo_base, pseudo_novel, pseudo_support, splits, vehicle_sets


def text_features(weights, tokenizer_path, class_names=TYPE_NAMES, templates=TEMPLATES):
    from openai_clip_model import build_model
    from transformers import CLIPTokenizer
    import torch.nn.functional as F

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    saved = torch.jit.load(str(weights), map_location="cpu")
    model = build_model(saved.state_dict()).cuda().eval()
    prompts = [template.format(name) for template in templates for name in class_names]
    # Match the original OpenAI CLIP tokenization: pad with zeros after EOT.
    encoded = tokenizer(prompts, padding=False)["input_ids"]
    tokens = torch.zeros((len(prompts), 77), device="cuda", dtype=torch.long)
    for row, ids in enumerate(encoded):
        tokens[row, :len(ids)] = torch.tensor(ids, device="cuda")
    anchors = F.normalize(model.encode_text(tokens).float(), dim=1)
    return F.normalize(anchors.reshape(len(templates), len(class_names), -1).mean(0), dim=1)


def fit_readout(features, labels, indices, base, tune, text, fscil, gda, moment_tune=None):
    tune_labels = labels[tune]
    best_correct = -1
    selected = None
    for regularization in (.01, .1, 1., 10.):
        ridge = fscil.ridge(features[indices], labels[indices], base, regularization)
        correct = fscil.prediction(fscil.lin(features[tune], ridge), base) == tune_labels
        acc = fscil.macro(correct, tune_labels, base)
        if acc > best_correct:
            best_correct = acc
            selected = (regularization, ridge)
    regularization, ridge = selected
    prototypes = fscil.means(features[indices], labels[indices], base)
    means, precision = gda.precision(features[indices], labels[indices], base)
    readout = (ridge, prototypes, means, precision)
    moment_tune = tune if moment_tune is None else moment_tune
    reference = raw_evidence(features[moment_tune], text[base], readout, prototypes, means, fscil, gda)
    moments = [
        (float(values.mean()), float(values.std(1, correction=0).mean().clamp_min(1e-6)))
        for values in reference
    ]
    return readout, moments, regularization


def raw_evidence(queries, text, readout, prototypes, means, fscil, gda):
    ridge, _, _, precision = readout
    return [queries @ text.T, fscil.lin(queries, ridge), queries @ prototypes.T,
            gda.lda(queries, means, precision)]


def align(raw, nb, moments):
    # Section 4.1 / Supplement K, used at both pseudo-expansion and deployment.
    eta = min(1., nb / 30.)
    aligned = []
    for values, (fixed_mean, fixed_sd) in zip(raw, moments):
        reference = values[:, :nb]
        mu = eta * reference.mean(1, keepdim=True) + (1 - eta) * fixed_mean
        sd = eta * reference.std(1, correction=0, keepdim=True).clamp_min(1e-6) + (1 - eta) * fixed_sd
        aligned.append((values - mu) / sd.clamp_min(1e-6))
    return aligned


def select_gda(vision, text, base, labels, alphas, fscil):
    return max(alphas, key=lambda alpha: (
        fscil.macro(fscil.prediction(text + alpha * vision, base) == labels, labels, base),
        -alpha,
    ))


def best_base_novel_offset(component, fscil):
    """Base-only pseudo-stage tuning for a baseline with fixed score weights."""
    candidates = np.unique(np.quantile(component["margin"], QUANTILES))
    best_harmonic, best = -1., None
    for offset in candidates:
        values = fscil.evaluate(component, offset)
        ab, an = values["base"], values["novel"]
        harmonic = 2 * ab * an / (ab + an) if ab + an else 0.
        if harmonic > best_harmonic:
            best_harmonic, best = harmonic, float(offset)
    return best, best_harmonic


def select_positive_prototype(evidence, base_selection, ids, labels, base, novel, fscil, anchor):
    """Refit the novel prototype weight and admission offset with a and g fixed."""
    candidates = []
    for u in fscil.GRID:
        if u <= 0:
            continue
        scores = anchor.fuse(evidence[:3], evidence[3], base_selection["a"], u,
                             base_selection["g"], len(base))
        component = fscil.comp(scores, ids, labels, base, novel)
        offset, harmonic = best_base_novel_offset(component, fscil)
        candidates.append(dict(base_selection, u=u, offset=offset, harmonic=harmonic))
    return max(candidates, key=lambda selection: (selection["harmonic"], -selection["u"]))


def apply_base_margin(scores, nb, coefficient):
    """Apply one new-class offset per query based on the top two base logits."""
    if not coefficient:
        return scores
    top_base = scores[:, :nb].topk(2, dim=1).values
    gap = top_base[:, :1] - top_base[:, 1:2]
    return torch.cat((scores[:, :nb], scores[:, nb:] - coefficient * gap), dim=1)


def select_base_margin(scores, ids, labels, base, novel, fscil):
    base_rows = np.isin(labels, base)
    reference = fscil.prediction(scores[base_rows, :len(base)], base)
    target = .95 * fscil.macro(reference == labels[base_rows], labels[base_rows], base)
    best = None
    for coefficient in fscil.GRID:
        adjusted = apply_base_margin(scores, len(base), coefficient)
        component = fscil.comp(adjusted, ids, labels, base, novel)
        frontier = fscil.frontier(component, target)
        if frontier["feasible"] and (best is None or frontier["novel"] > best["novel"] + 1e-12):
            best = {"coefficient": coefficient, "novel": frontier["novel"], "target_base": target,
                    "offset": best_base_novel_offset(component, fscil)[0]}
    if best is None:
        raise ValueError("No feasible pseudo-episode operating point")
    return best


def certify_bet_one(component, reference, weights, candidates, sessions, choose_new):
    """Fixed-sequence paired certification with the original grid's bet λ=1."""
    if np.any(component["base_correct"] & ~reference):
        raise ValueError("The fixed-bet variant expects the unchanged base winner")
    scaled_weights = weights / weights.max()
    trace, accepted = [], None
    for offset in candidates:
        correct = component["base_correct"] & ~choose_new(component, offset)
        surplus = correct.astype(float) - .95 * reference
        log_wealth = float(np.log1p(scaled_weights * surplus).sum())
        pvalue = float(np.exp(-max(0., log_wealth)))
        passed = pvalue <= .05 / sessions
        trace.append({"offset": float(offset), "pvalue": pvalue, "accepted": passed, "log_evalue": log_wealth,
                      "accuracy_surplus": float(weights @ surplus),
                      "destroyed": float(weights @ (reference & ~correct)),
                      "recovered": float(weights @ (~reference & correct))})
        if not passed:
            break
        accepted = float(offset)
    return {"offset": accepted, "trace": trace}


def dense_offsets(tuning_margins):
    """Build a fixed finer sequence and extend its conservative end."""
    coarse = np.quantile(tuning_margins, QUANTILES)
    dense = np.quantile(tuning_margins, np.linspace(0, 1, 1001))
    spread = max(float(coarse[75] - coarse[25]), 1e-6)
    lower = coarse[0] - spread * np.array([.01, .025, .05, .1, .2, .4, .8, 1.6])
    return np.unique(np.r_[lower, coarse, dense])


def teen_scores(queries, prototypes, nb, fscil):
    if len(prototypes) == nb:
        return queries @ prototypes.T
    base, novel = prototypes[:nb], prototypes[nb:]
    similarity = novel @ base.T
    neighbors = torch.softmax(fscil.TEEN_TAU * similarity, dim=1) @ base
    calibrated = fscil.norm(fscil.TEEN_ALPHA * novel + (1 - fscil.TEEN_ALPHA) * neighbors)
    return queries @ torch.cat([base, calibrated]).T


@torch.inference_mode()
def experiment(features, text, samples, seed, fscil, anchor, gda, require_prototype=False,
               compare_admission=False, dense_offsets_all=False, fscil_baselines=False,
               protocol_fn=protocol, calibration_caps=CALIBRATION_CAPS):
    labels, base, arrival, pb, pn, ps, splits, vehicle_sets = protocol_fn(samples, seed)
    fit, tune, cal, test = (splits[key] for key in ("fit", "tune", "cal", "test"))
    pfit = fit[np.isin(labels[fit], pb)]
    ptune = tune[np.isin(labels[tune], pb)]
    pseudo, pm, pseudo_lambda = fit_readout(features, labels, pfit, pb, ptune, text, fscil, gda, moment_tune=tune)
    pi = np.r_[pb, pn]
    pseudo_prototypes = torch.cat([pseudo[1], fscil.means(features[ps], labels[ps], pn)])
    pseudo_means = torch.cat([pseudo[2], anchor.raw_means(features[ps], labels[ps], pn)])
    pq = features[tune]
    p_evidence = align(raw_evidence(pq, text[pi], pseudo, pseudo_prototypes, pseudo_means, fscil, gda),
                       len(pb), pm)
    original = anchor.select(p_evidence[:3], p_evidence[3], pi, labels[tune], pb, pn)
    param = select_positive_prototype(p_evidence, original, pi, labels[tune], pb, pn, fscil, anchor) if require_prototype else original
    readout, moments, ridge_lambda = fit_readout(features, labels, fit, base, tune, text, fscil, gda)
    _, prototypes, means, precision = readout
    offsets = {"AnchorCE": param["offset"]}
    gda_parameters = {}
    if fscil_baselines:
        offsets.update({"HyCal": 0., "TEEN": 0.})
    else:
        tq = features[tune]
        base_text, base_cov = tq @ text[base].T, gda.lda(tq, means, precision)
        alpha = select_gda(base_cov, 100 * base_text, base, labels[tune], gda.ALPHAS, fscil)
        tune_aligned = align(raw_evidence(tq, text[base], readout, prototypes, means, fscil, gda),
                             len(base), moments)
        alpha_anchor = select_gda(tune_aligned[3], tune_aligned[0], base, labels[tune], fscil.GRID, fscil)
        p_train = np.r_[pfit, ps]
        p_sorted = np.sort(pi)
        updated_means, updated_precision = gda.precision(features[p_train], labels[p_train], p_sorted)
        p_gda = 100 * pq @ text[pi].T + alpha * gda.lda(pq, updated_means[np.searchsorted(p_sorted, pi)],
                                                      updated_precision)
        offsets["GDA"] = best_base_novel_offset(fscil.comp(p_gda, pi, labels[tune], pb, pn), fscil)[0]
        offsets["GDA-Anchor"] = best_base_novel_offset(
            fscil.comp(p_evidence[0] + alpha_anchor * p_evidence[3], pi, labels[tune], pb, pn), fscil)[0]
        gda_parameters = {"gda_alpha": alpha, "anchor_gda_alpha": alpha_anchor}
    gate_selection = None
    if compare_admission:
        pscores = anchor.fuse(p_evidence[:3], p_evidence[3], param["a"], param["u"], param["g"], len(pb))
        gate_selection = select_base_margin(pscores, pi, labels[tune], pb, pn, fscil)
        offsets["AnchorCE-BaseMargin"] = gate_selection["offset"]
        for name in ("AnchorCE-Bet1", "AnchorCE-DenseOffsets"):
            offsets[name] = param["offset"]
    methods = FSCIL_METHODS if fscil_baselines else METHODS
    if compare_admission:
        methods += ADMISSION_VARIANTS
    pred_cal, closed, frozen_scores = {}, {}, {}
    sessions = []
    for session in range(len(arrival) + 1):
        novel = np.sort(arrival[:session])
        active = np.r_[base, novel]
        chosen_support = splits["support"][np.isin(labels[splits["support"]], novel)]
        active_test = test[np.isin(labels[test], active)]
        queries = features[np.r_[tune, cal, active_test]]
        session_proto = torch.cat([prototypes, fscil.means(features[chosen_support], labels[chosen_support], novel)]) if session else prototypes
        session_means = torch.cat([means, anchor.raw_means(features[chosen_support], labels[chosen_support], novel)]) if session else means
        aligned = align(raw_evidence(queries, text[active], readout, session_proto, session_means, fscil, gda),
                        len(base), moments)
        expanded_fit = np.r_[fit, chosen_support]
        score_map = {"AnchorCE": anchor.fuse(aligned[:3], aligned[3], param["a"], param["u"], param["g"], len(base))}
        if fscil_baselines:
            score_map["HyCal"] = fscil.hycal(queries, text[active], features[expanded_fit], labels[expanded_fit], active)
            score_map["TEEN"] = teen_scores(queries, session_proto, len(base), fscil)
        else:
            sorted_active = np.sort(active)
            if session:
                mus, P = gda.precision(features[expanded_fit], labels[expanded_fit], sorted_active)
            else:
                mus, P = means, precision
            score_map["GDA"] = 100 * queries @ text[active].T + alpha * gda.lda(
                queries, mus[np.searchsorted(sorted_active, active)], P)
            score_map["GDA-Anchor"] = aligned[0] + alpha_anchor * aligned[3]
        if compare_admission:
            score_map["AnchorCE-BaseMargin"] = apply_base_margin(score_map["AnchorCE"], len(base),
                                                                 gate_selection["coefficient"])
            for name in ("AnchorCE-Bet1", "AnchorCE-DenseOffsets"):
                score_map[name] = score_map["AnchorCE"]
        session_result = {"session": session, "seen_novel": novel.tolist(), "methods": {}}
        end_tune, end_cal = len(tune), len(tune) + len(cal)
        for name in methods:
            scores = score_map[name]
            if not session:
                frozen_scores[name] = scores[:end_cal, :len(base)].clone()
                pred_cal[name] = fscil.prediction(scores[end_tune:end_cal], base) == labels[cal]
                test_pred = fscil.prediction(scores[end_cal:], base)
                closed[name] = fscil.macro(test_pred == labels[active_test], labels[active_test], base)
                session_result["methods"][name] = {"closed_base_macro": closed[name]}
                continue
            if name not in ("GDA", "HyCal"):
                torch.testing.assert_close(scores[:end_cal, :len(base)], frozen_scores[name], atol=1e-4, rtol=1e-5)
            tuning = fscil.comp(scores[:end_tune], active, labels[tune], base, novel)
            heldout = fscil.comp(scores[end_cal:], active, labels[active_test], base, novel)
            if dense_offsets_all or name == "AnchorCE-DenseOffsets":
                candidates = dense_offsets(tuning["margin"])
            else:
                candidates = np.unique(np.quantile(tuning["margin"], QUANTILES))
            record = {"unconstrained": fscil.evaluate(heldout, offsets[name]), "contracts": {}}
            for contract, reference_name in (("common", "AnchorCE"), ("self", name)):
                reference_base = closed[reference_name]
                certifications = {}
                for cap in calibration_caps:
                    sub = np.concatenate([np.flatnonzero(labels[cal] == k)[:cap] for k in base])
                    calibration = fscil.comp(scores[end_tune:end_cal][sub], active, labels[cal][sub], base, novel)
                    weights = fscil.risk.weights(labels[cal][sub], base)
                    reference = pred_cal[reference_name][sub]
                    if name == "AnchorCE-Bet1":
                        detail = certify_bet_one(calibration, reference, weights, candidates, len(arrival),
                                                 fscil.risk.choose_new)
                    else:
                        detail = fscil.paired.certify(calibration, reference, weights, candidates, sessions=len(arrival))
                    offset = detail["offset"]
                    metrics = {"base": reference_base, "novel": 0.,
                               "overall": reference_base * len(base) / len(active)} if offset is None else fscil.evaluate(heldout, offset)
                    chosen_correct = calibration["base_correct"] if offset is None else (
                        calibration["base_correct"] & ~fscil.risk.choose_new(calibration, offset))
                    recoveries = float(weights @ (~reference & chosen_correct))
                    if name == reference_name and name not in ("GDA", "HyCal"):
                        assert recoveries == 0
                    certifications[str(cap)] = {
                        **metrics, "admitted": offset is not None, "offset": offset,
                        "meets_test_target": metrics["base"] + 1e-10 >= .95 * reference_base,
                        "first_test": detail["trace"][0],
                        "closed_winner_recovery": float(weights @ (~reference & calibration["base_correct"])),
                    }
                record["contracts"][contract] = {
                    "reference": reference_name, "reference_base": reference_base,
                    "budgets": certifications, "oracle": fscil.frontier(heldout, .95 * reference_base),
                }
            session_result["methods"][name] = record
        sessions.append(session_result)
        print(f"seed={seed} session={session}/{len(arrival)} "
              + " ".join(f"{name}={row.get('unconstrained', {}).get('overall', row.get('closed_base_macro')):.4f}"
                         for name, row in session_result["methods"].items()), flush=True)
    return {"seed": seed, "base": base.tolist(), "arrival_order": arrival.tolist(),
            "pseudo_base": pb.tolist(), "anchor_selection": param, "ridge_lambda": ridge_lambda,
            **({"original_anchor_selection": original} if require_prototype else {}),
            **({"base_margin_selection": gate_selection} if compare_admission else {}),
            "pseudo_ridge_lambda": pseudo_lambda, **gda_parameters,
            "uncalibrated_offsets": offsets, "role_vehicle_ids": {name: sorted(ids) for name, ids in vehicle_sets.items()},
            "role_indices": {name: ix.tolist() for name, ix in splits.items()}, "pseudo_support_indices": ps.tolist(),
            "sessions": sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="JLT-Head autodl_fscil directory")
    parser.add_argument("--cache", type=Path, default=Path("results"))
    parser.add_argument("--weights", type=Path, default=Path("ViT-B-16.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("clip_tokenizer"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-prototype", action="store_true",
                        help="Select u > 0 on the pseudo episode while preserving the original a and g")
    parser.add_argument("--compare-admission", action="store_true",
                        help="Compare the margin gate, fixed λ=1, and denser admission offsets")
    parser.add_argument("--dense-offsets-all", action="store_true",
                        help="Use the same denser candidate construction for every scoring method")
    parser.add_argument("--fscil-baselines", action="store_true",
                        help="Use original-score HyCal and TEEN transfers as the baseline pair")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.compare_admission and not args.require_prototype:
        parser.error("--compare-admission expects --require-prototype")
    if args.output is None:
        args.output = Path("anchorce_fscil_baselines_results" if args.fscil_baselines else
                           "anchorce_matched_results" if args.dense_offsets_all else
                           "anchorce_admission_results" if args.compare_admission else
                           "anchorce_prototype_results" if args.require_prototype else "anchorce_results")
    sys.path.insert(0, str(args.source.resolve()))
    import anchor_gda as anchor
    import extra_baselines as gda
    import fscil_readouts as fscil

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)
    samples = json.loads((args.cache / "samples.json").read_text(encoding="utf-8"))
    feature_array = np.load(args.cache / "features.npy")
    assert feature_array.shape == (len(samples), 512) and np.isfinite(feature_array).all()
    features = torch.from_numpy(feature_array).cuda()
    with torch.inference_mode():
        text = text_features(args.weights, args.tokenizer)
    records = []
    for seed in args.seeds:
        result = experiment(features, text, samples, seed, fscil, anchor, gda, args.require_prototype,
                            args.compare_admission, args.dense_offsets_all, args.fscil_baselines)
        records.append(result)
        fscil.atomic(result, args.output / f"seed_{seed}.json")
    fscil.atomic({"dataset": "VehicleX VeRi_ReID_Simulation", "encoder": "OpenAI CLIP ViT-B/16",
                  "types": TYPE_NAMES, "ratio": .95, "delta": .05,
                  "calibration_caps": CALIBRATION_CAPS, "require_prototype": args.require_prototype,
                  "compare_admission": args.compare_admission, "dense_offsets_all": args.dense_offsets_all,
                  "fscil_baselines": args.fscil_baselines,
                  "runs": records}, args.output / "metrics.json")


if __name__ == "__main__":
    main()
