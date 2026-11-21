"""VehicleX attribute benchmark with vehicle-disjoint splits and frozen CLIP."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class Sample:
    image: str
    vehicle: str
    label: str


def inventory(archive, external_xml=()):
    """Join per-image annotations to ZIP members by unique basename."""
    images = {}
    duplicates = set()
    with ZipFile(archive) as data:
        for member in data.namelist():
            path = PurePosixPath(member)
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not member.startswith("__MACOSX/"):
                if path.name in images:
                    duplicates.add(path.name)
                images[path.name] = member
        for name in duplicates:
            del images[name]
        xml_files = [name for name in data.namelist() if name.lower().endswith(".xml")]
        xml_data = [(name, data.read(name)) for name in xml_files]
    xml_data.extend((str(path), Path(path).read_bytes()) for path in external_xml)
    attributes = {"type": {}, "color": {}}
    for name, content in xml_data:
        for node in ET.fromstring(content).iter():
            fields = {re.sub(r"\W|_", "", key).lower(): val.strip() for key, val in node.attrib.items()}
            image = fields.get("imagename") or fields.get("filename") or fields.get("image")
            if not image:
                continue
            image = PurePosixPath(image.replace("\\", "/")).name
            if image not in images:
                continue
            vehicle = fields.get("vehicleid") or fields.get("id") or image.split("_", 1)[0]
            for attribute in attributes:
                value = fields.get(attribute + "id") or fields.get("vehicle" + attribute) or fields.get(attribute)
                if value and value not in {"-1", "unknown"}:
                    sample = Sample(images[image], vehicle, value)
                    previous = attributes[attribute].setdefault(image, sample)
                    if previous != sample:
                        raise ValueError(f"Conflicting {attribute} annotations for {image} in {name}")
    summary = {"archive_images": len(images), "ambiguous_basenames": len(duplicates),
               "xml_sources": [name for name, _ in xml_data], "attributes": {}}
    for attribute, rows in attributes.items():
        rows = sorted(rows.values(), key=lambda sample: sample.image)
        attributes[attribute] = rows
        summary["attributes"][attribute] = {
            "images": dict(Counter(sample.label for sample in rows)),
            "vehicles": dict(Counter(label for _, label in {(row.vehicle, row.label) for row in rows})),
        }
    return attributes, summary


def select_samples(rows, seed, cap):
    grouped = defaultdict(list)
    for sample in rows:
        grouped[(sample.vehicle, sample.label)].append(sample)
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(grouped):
        items = grouped[key]
        indices = rng.choice(len(items), min(cap, len(items)), replace=False)
        selected.extend(items[index] for index in sorted(indices))
    return selected


def encode(rows, archive, weights, batch_size):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    from openai_clip_model import build_model

    transform = transforms.Compose([
        transforms.Resize(224, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    class VehicleImages(torch.utils.data.Dataset):
        def __init__(self):
            self.archive = None

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            if self.archive is None:
                self.archive = ZipFile(archive)
            with self.archive.open(rows[index].image) as file, Image.open(file) as image:
                return transform(image.convert("RGB"))

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    witness = torch.randn(64, 64, device="cuda")
    assert (witness @ witness).isfinite().all().item()
    print(f"CUDA witness passed on {torch.cuda.get_device_name()}", flush=True)
    saved = torch.jit.load(str(weights), map_location="cpu")
    model = build_model(saved.state_dict()).cuda().eval()
    del saved
    loader = torch.utils.data.DataLoader(VehicleImages(), batch_size=batch_size, num_workers=6, pin_memory=True)
    features = []
    with torch.inference_mode():
        for batch, images in enumerate(loader):
            encoded = model.encode_image(images.cuda(non_blocking=True))
            features.append(F.normalize(encoded.float(), dim=1).cpu().numpy())
            if batch % 25 == 0:
                print(f"Encoded {min((batch + 1) * batch_size, len(rows))}/{len(rows)} images", flush=True)
    features = np.concatenate(features)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite CLIP features")
    return features


def evaluate(features, rows, seeds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.array([sample.label for sample in rows])
    groups = np.array([sample.vehicle for sample in rows])
    classes = sorted(set(labels))
    runs = []
    for seed in seeds:
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train, test = next(folds.split(features, labels, groups))
        if set(labels[train]) != set(classes) or set(labels[test]) != set(classes):
            raise ValueError(f"Incomplete class coverage in the preset split for seed {seed}")
        assert not set(groups[train]).intersection(groups[test])
        frequencies = Counter(labels[train])
        n_tail = max(1, len(classes) // 3)
        tail = sorted(classes, key=lambda label: (frequencies[label], label))[:n_tail]
        rng = np.random.default_rng(seed)
        resampled = np.concatenate([
            rng.choice(train[labels[train] == label], max(frequencies.values()), replace=frequencies[label] < max(frequencies.values()))
            for label in classes
        ])
        run = {"seed": seed, "train_images": len(train), "test_images": len(test),
               "train_vehicles": len(set(groups[train])), "test_vehicles": len(set(groups[test])),
               "train_class_counts": dict(frequencies), "test_class_counts": dict(Counter(labels[test])),
               "tail_classes": tail, "methods": {}}
        for name, indices, weight in [("ordinary", train, None), ("reweighted", train, "balanced"),
                                      ("resampled", resampled, None)]:
            classifier = LogisticRegression(C=1.0, class_weight=weight, max_iter=600)
            classifier.fit(features[indices], labels[indices])
            prediction = classifier.predict(features[test])
            recalls = recall_score(labels[test], prediction, labels=classes, average=None, zero_division=0)
            per_class = dict(zip(classes, recalls.tolist()))
            run["methods"][name] = {
                "accuracy": accuracy_score(labels[test], prediction),
                "macro_f1": f1_score(labels[test], prediction, average="macro", zero_division=0),
                "macro_recall": float(np.mean(recalls)),
                "tail_recall": float(np.mean([per_class[label] for label in tail])),
                "per_class_recall": per_class,
            }
        runs.append(run)
        print(json.dumps(run, ensure_ascii=False), flush=True)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--xml", type=Path, nargs="*", default=[])
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--attribute", choices=["type", "color"], default="type")
    parser.add_argument("--weights", type=Path, default=Path("ViT-B-16.pt"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cap", type=int, default=24, help="maximum views per vehicle and attribute label")
    args = parser.parse_args()
    attributes, summary = inventory(args.archive, args.xml)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.inspect:
        return
    rows = select_samples(attributes[args.attribute], 2026, args.cap)
    if not rows:
        raise SystemExit(f"No matched {args.attribute} annotations to train on")
    args.output.mkdir(parents=True, exist_ok=True)
    sample_data = [vars(row) for row in rows]
    (args.output / "samples.json").write_text(json.dumps(sample_data, indent=2), encoding="utf-8")
    features = encode(rows, args.archive, args.weights, args.batch_size)
    np.save(args.output / "features.npy", features)
    runs = evaluate(features, rows, [42, 43, 44])
    result = {"attribute": args.attribute, "encoder": "OpenAI CLIP ViT-B/16", "views_per_vehicle_label_cap": args.cap,
              "inventory": summary, "runs": runs}
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
