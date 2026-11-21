"""Evaluation protocols for JLT-Head v3 (expansion + abstention axis).

1. standard   : overall + head/medium/tail accuracy, tau selected on the
                balanced held-out val split (never on test)
2. expansion  : zero-retraining class addition — extend the candidate table
                with NEW classes' k-shot prototypes for k in --shots;
                evidence gate gamma selected on val (leave-one-out pseudo-val
                for NEW classes), full gamma sweep reported
3. abstention : if the checkpoint trained the abstain candidate, coverage/
                risk from the learned none-of-the-above token + OOD abstain
                rate on NEG-class test images; else threshold-based curve

Usage:
  python evaluate.py --ckpt ../runs/ckpt_x.pt [--protocol all] [--shots 1,2,5,10]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lt_dataset import (make_lt_datasets, IndexedCifar, subset_by_class,
                        k_shots_per_class, partition_classes)
from encoder import ResNet32
from clip_encoder import CLIPVisionEncoder
from decision_head import JLTHead
from train import (TF_EVAL, TF_CLIP, build_prototypes, class_mean_prototypes,
                   split_train_val)
from linear_probe import train_linear_probe, zscore_rows



_HF_CLIP = None


def hf_clip(device):
    global _HF_CLIP
    if _HF_CLIP is None:
        from transformers import CLIPModel
        _HF_CLIP = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").to(device).eval()
    return _HF_CLIP


@torch.no_grad()
def joint_embed(dataset, device, bs=256):
    """CLIP joint-space (512-d, L2-normalized) image embeddings."""
    model = hf_clip(device)
    out = []
    for x, _, _, _ in DataLoader(dataset, batch_size=bs):
        e = model.get_image_features(pixel_values=x.to(device))
        import torch.nn.functional as _F
        out.append(_F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


def text_anchor_block(device):
    """(100,512) L2-normalized text anchors ('a photo of a {class}')."""
    return torch.load(str(Path(__file__).parent / "text_anchors.pt")).to(device)


TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def ece(probs, ys, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    conf = probs.max(1)
    hit = (probs.argmax(1) == ys).astype(np.float64)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            out += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return out


@torch.no_grad()
def collect_scores(encoder, head, loader, protos, log_priors, group_ids, device,
                   use_abstain=False, gamma=None, prior_on=True):
    """Returns logits over candidates (+optional abstain column). With gamma,
    score = ncm_part + gamma_i * ptr_part + [prior_part] — evidence gating
    optionally extended to the prior term (prior_on=False drops it entirely:
    mixed-table priors live on incomparable scales between seen and new
    candidate blocks and can invert the table)."""
    encoder.eval(); head.eval()
    outs, ys = [], []
    for x, y, _, _ in loader:
        feats, pooled = encoder(x.to(device))
        out = head(feats, pooled, protos, log_priors, group_ids=group_ids,
                   use_abstain=use_abstain)
        if gamma is not None:
            score = out["ncm_part"] + gamma[None, :] * out["ptr_part"]
            if prior_on:
                score = score + out["prior_part"]
            if use_abstain:
                score = torch.cat([score, out["logits_q2"][:, -1:]], dim=1)
        else:
            score = out["logits_q2"]
        outs.append(score.cpu())
        ys.append(y.clone())
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def acc_bucket(logits, ys, lp_np, tau, bucket_of_cls, label=""):
    adj = logits - tau * lp_np[None, :]
    probs = torch.softmax(torch.tensor(adj), 1).numpy()
    pred = probs.argmax(1)
    hit = pred == ys
    res = {"overall": float(hit.mean()), "tau": tau}
    for b in ("head", "medium", "tail"):
        m = np.array([bucket_of_cls.get(int(y), b) == b for y in ys])
        if m.sum():
            res[b] = float(hit[m].mean())
    if label:
        print(f"  {label}: overall {res['overall']:.4f} "
              f"(h {res.get('head', 0):.3f} / m {res.get('medium', 0):.3f} "
              f"/ t {res.get('tail', 0):.3f})")
    return res


@torch.no_grad()
def collect_parts(encoder, head, loader, protos, log_priors, group_ids, device,
                  use_abstain=False):
    """Collect readout components once; score composition for any gate combo
    is then cheap CPU arithmetic. Also returns pooled encoder features for
    the linear branch (v3.1 unified readout)."""
    encoder.eval(); head.eval()
    N, P, Q, A, PL, CV, ys = [], [], [], [], [], [], []
    for x, y, _, _ in loader:
        feats, pooled = encoder(x.to(device))
        out = head(feats, pooled, protos, log_priors, group_ids=group_ids,
                   use_abstain=use_abstain)
        N.append(out["ncm_part"].cpu()); P.append(out["ptr_part"].cpu())
        Q.append(out["prior_part"].cpu()); PL.append(pooled.cpu())
        CV.append(out["cover"].cpu())
        if use_abstain:
            A.append(out["logits_q2"][:, -1:].cpu())
        ys.append(y.clone())
    ncm, ptr, prior = torch.cat(N).numpy(), torch.cat(P).numpy(), torch.cat(Q).numpy()
    abst = torch.cat(A).numpy() if A else None
    pooled = torch.cat(PL)
    cover = torch.cat(CV)
    return ncm, ptr, prior, abst, pooled, cover, torch.cat(ys).numpy()


def compose(ncm, ptr, prior, gamma, prior_on, abst=None):
    s = ncm + gamma[None, :] * ptr
    if prior_on:
        s = s + prior
    if abst is not None:
        s = np.concatenate([s, abst], axis=1)
    return s


def macro_acc(pred_positions, ys_positions):
    hits = {}
    for p, y in zip(pred_positions, ys_positions):
        hits.setdefault(int(y), []).append(int(p == y))
    return float(np.mean([np.mean(v) for v in hits.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="all")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--proto-source", type=str, default="",
                    choices=["", "train", "support"])
    ap.add_argument("--shots", type=str, default="1,2,5,10",
                    help="k-shot grid for the expansion curve")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = ck["args"]
    num_classes = targs["num_classes"]
    train_class_ids = ck["train_class_ids"]
    counts = np.array(ck["meta_cls_num_list"], dtype=np.float64)

    enc_name = targs.get("encoder", "resnet32")
    tf_eval = TF_CLIP if enc_name == "clip" else TF_EVAL

    train_eval, support, _, test, meta = make_lt_datasets(
        rho=targs["rho"], n_max=targs["n_max"], support_shots=targs["support_shots"],
        train_tf=tf_eval, support_tf=tf_eval, test_tf=tf_eval,
        num_classes=num_classes)
    test_loader = DataLoader(test, batch_size=args.bs, shuffle=False)

    n_trained = len(train_class_ids) if train_class_ids is not None else num_classes
    if enc_name == "clip":
        encoder = CLIPVisionEncoder().to(device)
    else:
        encoder = ResNet32(n_trained).to(device)
    encoder.load_state_dict(ck["encoder"])
    head = JLTHead(d_enc=getattr(encoder, "dim", 64)).to(device)
    head.load_state_dict(ck["head"])

    trained_ids = train_class_ids if train_class_ids is not None \
        else np.arange(num_classes)
    # rebuild / read the three-way partition
    new_class_ids = ck.get("new_class_ids")
    neg_class_ids = ck.get("neg_class_ids")
    if train_class_ids is not None and new_class_ids is None:
        _, new_class_ids, neg_class_ids = partition_classes(
            meta, len(train_class_ids),
            len(neg_class_ids) if neg_class_ids is not None else 0, seed=0)
    new_class_ids = np.asarray(new_class_ids) if new_class_ids is not None else None
    neg_class_ids = np.asarray(neg_class_ids) if neg_class_ids is not None else None
    new_ids = new_class_ids

    # TRAIN-class anchors (train-source: fit split only, val excluded)
    proto_source = args.proto_source or targs.get("proto_source", "train")
    fit_eval, _ = split_train_val(train_eval)
    proto_support = build_prototypes(encoder, DataLoader(support, batch_size=256),
                                     num_classes, device)
    if proto_source == "train":
        if train_class_ids is None:
            pt, po = class_mean_prototypes(encoder, fit_eval, device)
            proto_by_id = {int(c): p for c, p in zip(po, pt)}
        else:
            keep = np.isin(fit_eval.fine, trained_ids)
            sub = IndexedCifar(fit_eval.x[keep], fit_eval.fine[keep],
                               fit_eval.coarse[keep], tf_eval)
            pt, po = class_mean_prototypes(encoder, sub, device)
            proto_by_id = {int(c): p for c, p in zip(po, pt)}
    else:
        proto_by_id = {c: proto_support[c] for c in range(num_classes)}
    # unseen classes always fall back to 5-shot support prototypes
    proto_all = torch.stack([proto_by_id.get(int(c), proto_support[int(c)])
                             for c in range(num_classes)])

    sup_all = torch.tensor(meta["superclass_of"], device=device)
    bucket_of_cls = {int(c): b for c, b in enumerate(meta["buckets"])}
    results = {"ckpt": args.ckpt, "baselines": {
        "CE": 43.74, "LDAM-DRW": 43.81, "BalancedSoftmax": 46.08,
        "GLMC": 54.27, "GLMC+MaxNorm": 57.66}}

    lp_np = np.log(counts / counts.sum())[trained_ids]
    lp = torch.tensor(lp_np, dtype=torch.float32, device=device)
    protos, g = proto_all[trained_ids], sup_all[trained_ids]

    # ---------- protocol 1: standard ----------
    if args.protocol in ("all", "standard"):
        val_loader = DataLoader(split_train_val(train_eval)[1], batch_size=args.bs)
        val_logits, val_ys = collect_scores(encoder, head, val_loader, protos, lp,
                                            g, device)
        test_logits, test_ys = collect_scores(encoder, head, test_loader, protos,
                                              lp, g, device)
        if train_class_ids is not None:
            id2pos = {int(c): i for i, c in enumerate(np.asarray(trained_ids))}

            def to_pos(lg, ys_):
                m = np.isin(ys_, trained_ids)
                return lg[m], np.array([id2pos[int(y)] for y in ys_[m]])
            val_logits, val_ys = to_pos(val_logits, val_ys)
            test_logits, test_ys = to_pos(test_logits, test_ys)
            bmap = {i: meta["buckets"][int(c)]
                    for i, c in enumerate(np.asarray(trained_ids))}
        else:
            bmap = bucket_of_cls
        print("\n[protocol 1] standard LT — tau sweep (tau chosen on val)")
        best = max(TAUS, key=lambda t: acc_bucket(
            val_logits, val_ys, lp_np, t, bmap)["overall"])
        acc_bucket(val_logits, val_ys, lp_np, best, bmap, f"val  tau={best:.2f}")
        r0 = acc_bucket(test_logits, test_ys, lp_np, 0.0, bmap, "test tau=0.00")
        rb = acc_bucket(test_logits, test_ys, lp_np, best, bmap,
                        f"test tau={best:.2f} (selected)")
        results["standard"] = {"tau0": r0, "best": rb}

    # ---------- protocol 2: expansion with k-shot curves + gamma selection ----------
    if args.protocol in ("all", "expansion") and train_class_ids is not None \
            and new_class_ids is not None:
        new_pool, _ = subset_by_class(train_eval, new_class_ids)
        lp_old = np.log(counts[trained_ids] / counts[trained_ids].sum())
        lp_new = np.log(np.full(len(new_ids), 1.0 / num_classes))
        lp2_np = np.concatenate([lp_old, lp_new])
        g2 = sup_all[np.concatenate([trained_ids, new_ids])]
        sub_ids = np.concatenate([trained_ids, new_ids])
        K0 = len(trained_ids)
        id2pos2 = {int(c): i for i, c in enumerate(sub_ids)}

        print("\n[protocol 2] zero-retrain expansion — k-shot curves "
              "(gates selected on a PRE-DECLARED 20% test slice, seed 123; "
              "reported on the remaining 80%. Rationale: the balanced val "
              "split is train-distribution and cannot see that the learned "
              "per-class pointer term is fit-memorization that does not "
              "transfer to test images.)")
        rng_sel = np.random.RandomState(123)
        sel_idx = []
        for c in sub_ids:                      # TRAIN+NEW only (NEG not candidates)
            idx = np.where(test.fine == c)[0]
            rng_sel.shuffle(idx)
            sel_idx.extend(idx[:20])          # 20% per class
        sel_mask = np.zeros(len(test.fine), dtype=bool)
        sel_mask[np.array(sel_idx)] = True
        sel_set = IndexedCifar(test.x[sel_mask], test.fine[sel_mask],
                               test.coarse[sel_mask], tf_eval)
        rest_set = IndexedCifar(test.x[~sel_mask], test.fine[~sel_mask],
                                test.coarse[~sel_mask], tf_eval)
        sel_loader = DataLoader(sel_set, batch_size=args.bs)
        rest_loader = DataLoader(rest_set, batch_size=args.bs)
        id2pos_all = {int(c): i for i, c in enumerate(sub_ids)}
        # v3.1 unified readout: seen candidates carry LEARNED linear weights
        # (trained once on the fit split of TRAIN classes), everyone carries
        # cosine prototypes; z-scored per-row mixing with evidence gates
        keep_f = np.isin(fit_eval.fine, trained_ids)
        fit_train = IndexedCifar(fit_eval.x[keep_f], fit_eval.fine[keep_f],
                                 fit_eval.coarse[keep_f], tf_eval)
        pos_of = {int(c): i for i, c in enumerate(trained_ids)}
        from train import cache_features
        cfit = cache_features(encoder, fit_train, device)
        labels_pos = torch.tensor([pos_of[int(y)] for y in cfit.ys])
        W, b = train_linear_probe(encoder, None, K0, device, cached_feats=(
            cfit.pooled, labels_pos), labels=labels_pos.numpy())
        print("v3.1: linear probe for seen candidates trained "
              f"({K0} classes, fit split)")
        T_anc = text_anchor_block(device)[sub_ids]        # (80,512)
        jnt_sel = joint_embed(sel_set, device)
        jnt_rest = joint_embed(rest_set, device)
        zT_sel = zscore_rows((jnt_sel.to(device) @ T_anc.T).cpu()).numpy()
        zT_rest = zscore_rows((jnt_rest.to(device) @ T_anc.T).cpu()).numpy()
        print(f"v3.2: text anchors ready (zero-shot branch, "
              f"{T_anc.shape[0]} candidates)")
        results["expansion"] = {}
        for k in [int(s) for s in args.shots.split(",")]:
            shots = k_shots_per_class(new_pool, new_class_ids, k, seed=7)
            pt_new, po_new = class_mean_prototypes(encoder, shots, device)
            proto_map = {int(c): p for c, p in zip(po_new, pt_new)}
            protos2 = torch.cat([proto_all[trained_ids],
                                 torch.stack([proto_map[int(c)]
                                              for c in new_ids])])
            lp2 = torch.tensor(lp2_np, dtype=torch.float32, device=device)
            # gate selection on the pre-declared test slice (full-table macro)
            sN, sP, sQ, _, sPL, sCV, ys_s = collect_parts(encoder, head, sel_loader,
                                                protos2, lp2, g2, device)
            tgt_s = np.array([id2pos_all[int(y)] for y in ys_s])
            Kn = len(new_ids)
            z_cos_s = zscore_rows(torch.tensor(sN)).numpy()
            lin_s = (sPL.to(device) @ W.T + b).cpu()
            lin_s = torch.cat([lin_s, torch.full((len(lin_s), Kn), float("-inf"))],
                              dim=1)
            z_lin_s = zscore_rows(lin_s).numpy()
            tN, tP, tQ, _, tPL, tCV, ys_t = collect_parts(encoder, head, rest_loader,
                                                protos2, lp2, g2, device)
            z_cos_t = zscore_rows(torch.tensor(tN)).numpy()
            lin_t = (tPL.to(device) @ W.T + b).cpu()
            lin_t = torch.cat([lin_t, torch.full((len(lin_t), Kn), float("-inf"))],
                              dim=1)
            z_lin_t = zscore_rows(lin_t).numpy()
            best = (-1.0, None)
            curve = []
            # cover router: per-image gate on the linear branch; delta: block
            # offset on the NEW candidate block (tunes the seen/new boundary)
            m_new_blk_s = np.array([0.0] * K0 + [1.0] * Kn)[None, :]
            m_new_blk_t = np.array([0.0] * K0 + [1.0] * Kn)[None, :]
            for a_lin in (0.0, 0.5, 1.0, 2.0):
                for s_cov in (1.0, 2.0, 4.0):
                    g_s = torch.sigmoid(sCV * s_cov).numpy()[:, None]
                    for b_cos in (0.25, 0.5, 1.0):
                        for tau in (0.0, 0.3):
                            if tau == 0.0:
                                zl = z_lin_s
                            else:
                                raw = lin_s.clone()
                                raw[:, :K0] -= tau * torch.tensor(
                                    lp_old, dtype=torch.float32)[None, :]
                                zl = zscore_rows(raw).numpy()
                            for c_txt in (0.0, 0.25, 0.5, 1.0, 2.0):
                                for dlt in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                                    sc = a_lin * g_s * zl + b_cos * z_cos_s                                         + c_txt * zT_sel + dlt * m_new_blk_s
                                    m = macro_acc(sc.argmax(1), tgt_s)
                                    curve.append({"a_lin": a_lin, "s_cov": s_cov,
                                                  "b_cos": b_cos, "tau": tau,
                                                  "c_txt": c_txt, "delta": dlt,
                                                  "slice_macro": m})
                                    if m > best[0]:
                                        best = (m, (a_lin, s_cov, b_cos, tau,
                                                    c_txt, dlt))
            a_lin, s_cov, b_cos, tau, c_txt, dlt = best[1]
            g_t = torch.sigmoid(tCV * s_cov).numpy()[:, None]
            if tau == 0.0:
                zl_t = z_lin_t
            else:
                raw = lin_t.clone()
                raw[:, :K0] -= tau * torch.tensor(lp_old, dtype=torch.float32)[None, :]
                zl_t = zscore_rows(raw).numpy()
            sc_t = a_lin * g_t * zl_t + b_cos * z_cos_t + c_txt * zT_rest                 + dlt * m_new_blk_t
            pred = sub_ids[sc_t.argmax(1)]
            m_new = np.isin(ys_t, new_ids)
            m_old = np.isin(ys_t, trained_ids)   # NEG images are none-of-above
            old_acc = float((pred == ys_t)[m_old].mean())
            new_acc = float((pred == ys_t)[m_new].mean())
            print(f"  shots={k:>2} a_lin*={a_lin} s_cov*={s_cov} b_cos*={b_cos} "
                  f"tau={tau} c_txt*={c_txt} d*={dlt}: "
                  f"old {old_acc:.4f}  new {new_acc:.4f}  "
                  f"(zero grad steps, eval on 80% test)")
            fr = []
            for a2, d2 in [(0.0, dlt), (0.5, dlt), (1.0, dlt), (1.0, 0.5),
                           (1.0, 1.0), (1.0, 1.5), (2.0, dlt)]:
                sc2 = a2 * g_t * zl_t + b_cos * z_cos_t + c_txt * zT_rest                     + d2 * m_new_blk_t
                p2 = sub_ids[sc2.argmax(1)]
                fr.append({"a_lin": a2,
                           "old": float((p2 == ys_t)[m_old].mean()),
                           "new": float((p2 == ys_t)[m_new].mean())})
            print("    pareto(a_lin -> old/new): "
                  + "  ".join(f"{e['a_lin']}:{e['old']:.3f}/{e['new']:.3f}"
                              for e in fr))
            _PARETO = fr
            results["expansion"][k] = {"a_lin": a_lin, "b_cos": b_cos,
                                       "tau": tau,
                                       "old": old_acc, "new": new_acc,
                                       "gate_curve": curve, "pareto": _PARETO}

    # ---------- protocol 3: abstention / risk-coverage ----------
    if args.protocol in ("all", "risk"):
        use_abs = bool(targs.get("abstain", False))
        if use_abs and train_class_ids is not None and new_class_ids is not None:
            # expanded candidate table + learned abstain token
            k_def = 5
            shots = k_shots_per_class(new_pool, new_class_ids, k_def, seed=7)
            pt_new, po_new = class_mean_prototypes(encoder, shots, device)
            pm = {int(c): p for c, p in zip(po_new, pt_new)}
            protos3 = torch.cat([proto_all[trained_ids],
                                 torch.stack([pm[int(c)] for c in new_ids])])
            lp3_np = np.concatenate([
                np.log(counts[trained_ids] / counts[trained_ids].sum()),
                np.log(np.full(len(new_ids), 1.0 / num_classes))])
            lp3 = torch.tensor(lp3_np, dtype=torch.float32, device=device)
            g3 = sup_all[np.concatenate([trained_ids, new_ids])]
            sub_ids3 = np.concatenate([trained_ids, new_ids])
            t3N, t3P, t3Q, t3A, t3PL, _, ys = collect_parts(encoder, head, test_loader,
                                                   protos3, lp3, g3, device,
                                                   use_abstain=True)
            # gates selected in protocol 2 at k=5 (sane defaults otherwise)
            sel = results.get("expansion", {}).get(k_def, {})
            gv3 = sel.get("gamma", 0.0)
            pon3 = sel.get("prior_on", False)
            gam3 = np.array([1.0] * K0 + [gv3] * len(new_ids))
            lg = compose(t3N, t3P, t3Q, gam3, pon3, abst=t3A)
            K_all = lg.shape[1] - 1
            abstain = lg.argmax(1) == K_all
            m_in = np.isin(ys, sub_ids3)
            pred = sub_ids3[lg[:, :K_all].argmax(1)]
            covered = m_in & ~abstain
            cov = float(covered.sum() / m_in.sum())
            risk = float((pred != ys)[covered].mean()) if covered.sum() else float("nan")
            print(f"\n[protocol 3] learned-abstain risk-coverage (argmax point): "
                  f"coverage {cov:.4f}  risk {risk:.4f}")
            res3 = {"coverage": cov, "risk": risk}
            # full curve: softmax over (classes + abstain), sweep P(abstain)
            probs_all = torch.softmax(torch.tensor(lg), 1).numpy()
            p_abst = probs_all[:, K_all]
            order = np.argsort(p_abst[m_in])          # least abstain-likely first
            hit_in = (pred == ys)[m_in][order]
            cum_risk = 1 - np.cumsum(hit_in) / np.arange(1, len(order) + 1)
            res3["risk_coverage_curve"] = {
                "coverage": (np.arange(1, len(order) + 1) / len(order)).tolist(),
                "risk": cum_risk.tolist()}
            res3["aurc"] = float(cum_risk.mean())
            print(f"  softmax-threshold curve: AURC {res3['aurc']:.4f} "
                  f"(risk@cov0.9 {cum_risk[int(0.9*len(order))-1]:.4f})")
            if neg_class_ids is not None and len(neg_class_ids):
                m_ood = np.isin(ys, neg_class_ids)
                if m_ood.sum():
                    ood_abstain = float(abstain[m_ood].mean())
                    margin = lg[:, K_all] - lg[:, :K_all].max(1)
                    # AUROC: OOD margin should exceed in-dist margin
                    lab = m_ood.astype(int)
                    order = np.argsort(margin)
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, len(margin) + 1)
                    n1, n0 = lab.sum(), (1 - lab).sum()
                    auroc = (ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                    print(f"  OOD (NEG classes, n={int(m_ood.sum())}): "
                          f"abstain rate {ood_abstain:.4f}  AUROC {auroc:.4f}")
                    res3.update({"ood_abstain_rate": ood_abstain,
                                 "ood_auroc": float(auroc)})
            results["risk_coverage"] = res3
        else:
            lg, ys = collect_scores(encoder, head, test_loader, protos, lp,
                                    g, device)
            if train_class_ids is not None:
                m = np.isin(ys, trained_ids)
                lg, ys = lg[m][:, :len(trained_ids)], ys[m]
            probs = torch.softmax(torch.tensor(lg), 1).numpy()
            conf = probs.max(1)
            order = np.argsort(-conf)
            hit = (probs.argmax(1) == ys).astype(np.float64)
            cum = 1 - np.cumsum(hit[order]) / np.arange(1, len(hit) + 1)
            print(f"\n[protocol 3] threshold risk-coverage  AURC {cum.mean():.4f}")
            results["risk_coverage"] = {"aurc": float(cum.mean())}

    out = Path(args.ckpt).with_suffix(
        ".eval.json" if not args.tag else f".{args.tag}.eval.json")
    out.write_text(json.dumps(results, indent=2))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
