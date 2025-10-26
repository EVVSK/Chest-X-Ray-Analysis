import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def scan_split(root: Path, split_name: str):
    split_dir = root / split_name
    items = []
    if not split_dir.exists():
        return items
    has_class_subdirs = any(d.is_dir() for d in split_dir.iterdir())
    if has_class_subdirs:
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            label = cls_dir.name
            for p in cls_dir.rglob("*"):
                if p.is_file() and is_image_file(p):
                    items.append({"split": split_name, "label": label, "path": str(p)})
    else:
        for p in split_dir.rglob("*"):
            if p.is_file() and is_image_file(p):
                items.append({"split": split_name, "label": "_unknown", "path": str(p)})
    return items

def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--train", default="train")
    ap.add_argument("--test", default="test")
    ap.add_argument("--outDir", default=".")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.outDir)

    items = []
    items += scan_split(root, args.train)
    items += scan_split(root, args.test)

    if not items:
        print("No images found.")
        return

    # dataset_index.csv
    idx_rows = [
        {"split": r["split"], "label": r["label"], "path": r["path"]}
        for r in items
    ]
    write_csv(out_dir / "dataset_index.csv", idx_rows, ["split", "label", "path"])

    # class_counts.csv (wide format)
    labels = sorted({r["label"] for r in items})
    splits = sorted({r["split"] for r in items})
    counts = {s: {l: 0 for l in labels} for s in splits}
    for r in items:
        counts[r["split"]][r["label"]] = counts[r["split"]].get(r["label"], 0) + 1
    header = ["split"] + labels
    class_rows = []
    for s in splits:
        row = {"split": s}
        row.update(counts[s])
        class_rows.append(row)
    write_csv(out_dir / "class_counts.csv", class_rows, header)

    # proposed_train_val_split.csv
    # Assign 'val' for a stratified portion of train; keep others as their original split
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for i, r in enumerate(items):
        if r["split"] == "train":
            by_label[r["label"]].append(i)
    proposed_subset = [None] * len(items)
    for label, idxs in by_label.items():
        n = len(idxs)
        n_val = int(round(n * args.val_ratio))
        if n_val == 0 and n > 1:
            n_val = 1
        rng.shuffle(idxs)
        val_set = set(idxs[:n_val])
        for i in idxs:
            proposed_subset[i] = "val" if i in val_set else "train"
    # For non-train items, keep their split name
    for i, r in enumerate(items):
        if r["split"] != "train":
            proposed_subset[i] = r["split"]
    prop_rows = []
    for r, ps in zip(items, proposed_subset):
        row = {"split": r["split"], "label": r["label"], "path": r["path"], "proposed_subset": ps}
        prop_rows.append(row)
    write_csv(out_dir / "proposed_train_val_split.csv", prop_rows, ["split", "label", "path", "proposed_subset"])

    print("Wrote:")
    print(" -", out_dir / "dataset_index.csv")
    print(" -", out_dir / "class_counts.csv")
    print(" -", out_dir / "proposed_train_val_split.csv")

if __name__ == "__main__":
    main()
