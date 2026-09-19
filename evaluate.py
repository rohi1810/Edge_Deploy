"""Evaluate a trained PointNet++ checkpoint on the held-out TEST split.

Unlike train.py's validation split (used for early-stopping decisions during
training, so not a fully unbiased estimate), the test split is never touched
until this script runs -- the only honest read of real performance.

    python3 evaluate.py

Reports:
- Chamfer reconstruction distance (same metric train.py logs for val, so
  directly comparable to the val numbers printed during training)
- Per-point classification: overall accuracy (inflated by ~99% environment
  points, reported for context only) and per-class recall/precision/F1, which
  is the metric that actually matters given the extreme class imbalance
  (see README "The data").

--class-conf-threshold applies the same display rule the ROS2 inference path
uses (_shared_inference.ModelRunner): a point keeps its predicted object class
only if softmax confidence clears the threshold, otherwise it reads as
environment. 0 reproduces plain argmax. This is NOT a change to the weights --
it is a decision about what to show, and the sweep it exists to support is why
the demo states 0.97 rather than a number picked by eye:

    threshold   0.0     0.70    0.97
    macro-F1    0.681   0.787   0.867

--json-out writes the same numbers as machine-readable JSON, which is what the
demo's metrics payload is built from, so the page cannot drift from the script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PointCloudDataset
from losses import chamfer_distance
from models.pointnet2 import NUM_CLASSES, PointNet2MultiTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", type=Path,
                     default=Path(__file__).parent / "scenario_data" / "mbut_76scenarios" / "o_robot_nav")
    ap.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / "data" / "sim")
    ap.add_argument("--checkpoint", type=Path,
                     default=Path(__file__).parent / "outputs" / "pointnet2" / "best_multitask_model.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--class-conf-threshold", type=float, default=0.0,
                     help="object classes below this softmax confidence read as "
                          "environment; 0 = plain argmax (see module docstring)")
    ap.add_argument("--json-out", type=Path,
                     help="also write these numbers as JSON (used by demo/build_demo.py)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = np.load(args.splits_dir / "test_ids.npy")
    print(f"test: {len(test_ids)} samples, device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    loader = DataLoader(
        PointCloudDataset(args.recordings_dir, test_ids),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = PointNet2MultiTask().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.int64)
    recon_total, n_samples = 0.0, 0

    with torch.no_grad():
        for xyz, cls in loader:
            xyz, cls = xyz.to(device), cls.to(device)
            out = model(xyz)

            recon_total += chamfer_distance(out["recon_points"], xyz).item() * xyz.size(0)
            n_samples += xyz.size(0)

            logits = out["class_logits"]
            pred = logits.argmax(dim=-1)
            if args.class_conf_threshold > 0:
                conf = logits.softmax(dim=-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
                # Only object predictions are suppressed. Demoting a low-confidence
                # *environment* prediction to environment would be a no-op, and
                # promoting it to an object would invent detections.
                pred = torch.where((pred > 0) & (conf < args.class_conf_threshold),
                                   torch.zeros_like(pred), pred)
            idx = (cls.reshape(-1) * NUM_CLASSES + pred.reshape(-1)).cpu()
            confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)

    confusion = confusion.reshape(NUM_CLASSES, NUM_CLASSES)  # [true, pred]

    print(f"\nTest Chamfer reconstruction distance: {recon_total / n_samples:.5f}  "
          f"(val was ~0.924 at the restored checkpoint)")

    print(f"\nclass confidence threshold: {args.class_conf_threshold:g}"
          f"{'  (plain argmax)' if args.class_conf_threshold <= 0 else ''}")
    print(f"\n{'class':<12} {'true points':>12} {'recall':>8} {'precision':>10} {'f1':>8}")
    total_points = confusion.sum().item()
    correct = confusion.diag().sum().item()

    rows = []
    for c in range(NUM_CLASSES):
        true_count = confusion[c].sum().item()
        pred_count = confusion[:, c].sum().item()
        if true_count == 0 and pred_count == 0:
            continue
        recall = confusion[c, c].item() / true_count if true_count > 0 else float("nan")
        precision = confusion[c, c].item() / pred_count if pred_count > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        print(f"{CLASS_NAMES[c]:<12} {true_count:>12} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f}")
        rows.append({"classId": c, "name": CLASS_NAMES[c], "truePoints": true_count,
                     "recall": recall, "precision": precision, "f1": f1})

    # Macro-F1 over the OBJECT classes only. Including environment would fold a
    # ~0.999 F1 on 99% of the points into the average and hide every change the
    # threshold actually makes -- which is the whole quantity being optimised.
    object_rows = [r for r in rows if r["classId"] > 0]
    macro_f1 = sum(r["f1"] for r in object_rows) / len(object_rows) if object_rows else 0.0
    print(f"\nMacro-F1 over object classes: {macro_f1:.3f}")

    print(f"\nOverall point accuracy: {correct / total_points:.2%} "
          f"(inflated by ~99% environment points -- per-class recall above is the real signal)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "script": "evaluate.py",
            "checkpoint": str(args.checkpoint),
            "testSamples": int(len(test_ids)),
            "threshold": args.class_conf_threshold,
            "chamfer": recon_total / n_samples,
            "overallPointAcc": correct / total_points,
            "macroF1": macro_f1,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
