"""
run_inference_timing.py — Inference latency & parameter count
-------------------------------------------------------------
Mengukur ms/clip dan throughput (clips/sec) untuk tiga kondisi:
  1. swin3d_t — no-TD   (backbone ringan)
  2. swin3d_s — no-TD   (backbone terpilih, baseline)
  3. swin3d_s — td_gru  (proposed)

Input dummy: [batch, 3, T=16, 224, 224] di CUDA (jika tersedia).
Warmup 20 iterasi, ukur 100 iterasi.

Penggunaan:
  python run_inference_timing.py
  python run_inference_timing.py --batch_size 4

Output:
  Tabel ke stdout
  results/inference_timing.txt
"""

import argparse
import sys
import time
from pathlib import Path

import torch

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from model import VideoSwinFER


CONFIGS = [
    {
        "name":       "swin3d_t  no-TD",
        "backbone":   "swin3d_t",
        "use_td":     False,
        "td_variant": None,
        "td_config":  {},
    },
    {
        "name":       "swin3d_s  no-TD (baseline)",
        "backbone":   "swin3d_s",
        "use_td":     False,
        "td_variant": None,
        "td_config":  {},
    },
    {
        "name":       "swin3d_s  td_gru (proposed)",
        "backbone":   "swin3d_s",
        "use_td":     True,
        "td_variant": "td_gru",
        "td_config":  {
            "encoder_type": "gru",
            "num_layers":   1,
            "use_pos_enc":  False,
            "dim_ffn":      1536,
            "pool_type":    "mean",
        },
    },
]


def count_params(model: torch.nn.Module) -> dict:
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Backbone only (features + norm)
    bb_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("features") or name.startswith("norm")
    )
    return {"total_M": total / 1e6, "trainable_M": train / 1e6, "backbone_M": bb_params / 1e6}


def measure_latency(model: torch.nn.Module, dummy: torch.Tensor,
                    warmup: int = 20, reps: int = 100,
                    use_cuda: bool = True) -> dict:
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(reps):
            if use_cuda:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)   # ms

    import statistics
    mean_ms = statistics.mean(times)
    std_ms  = statistics.stdev(times)
    return {
        "mean_ms":    mean_ms,
        "std_ms":     std_ms,
        "fps":        1000.0 / mean_ms * dummy.shape[0],  # clips/sec
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_frames",  type=int, default=16)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--warmup",      type=int, default=20)
    p.add_argument("--reps",        type=int, default=100)
    args = p.parse_args()

    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device : {device}")
    print(f"Input  : [{args.batch_size}, 3, {args.num_frames}, {args.img_size}, {args.img_size}]")
    print(f"Warmup : {args.warmup} | Reps: {args.reps}\n")

    dummy = torch.randn(
        args.batch_size, 3, args.num_frames, args.img_size, args.img_size,
        device=device
    )

    rows = []
    for cfg in CONFIGS:
        model = VideoSwinFER(
            backbone    = cfg["backbone"],
            num_classes = args.num_classes,
            use_td      = cfg["use_td"],
            td_config   = cfg["td_config"] if cfg["use_td"] else None,
        ).to(device)

        params  = count_params(model)
        latency = measure_latency(model, dummy, args.warmup, args.reps, use_cuda)

        rows.append({
            "name":       cfg["name"],
            "total_M":    params["total_M"],
            "backbone_M": params["backbone_M"],
            "mean_ms":    latency["mean_ms"],
            "std_ms":     latency["std_ms"],
            "fps":        latency["fps"],
        })
        print(f"[✓] {cfg['name']}")

    # ── Print table ──
    header = f"{'Model':<34} {'Params(M)':>10} {'Backbone(M)':>12} {'ms/clip':>10} {'±':>6} {'clips/s':>9}"
    divider = "─" * len(header)
    lines = [
        "=" * len(header),
        "Inference Timing — iMED-VST",
        f"Device: {device} | Batch: {args.batch_size} | T={args.num_frames} | {args.img_size}px",
        "=" * len(header),
        header,
        divider,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<34} {r['total_M']:>10.1f} {r['backbone_M']:>12.1f} "
            f"{r['mean_ms']:>10.2f} {r['std_ms']:>6.2f} {r['fps']:>9.1f}"
        )
    lines += [divider]
    # Overhead td_gru vs no-TD baseline
    base = next(r for r in rows if "no-TD (baseline)" in r["name"])
    prop = next(r for r in rows if "td_gru" in r["name"])
    overhead_ms  = prop["mean_ms"] - base["mean_ms"]
    overhead_pct = overhead_ms / base["mean_ms"] * 100
    td_params    = prop["total_M"] - base["total_M"]
    lines.append(f"TD overhead: +{overhead_ms:.2f} ms/clip ({overhead_pct:+.1f}%)  |  TD params: +{td_params:.1f}M")
    lines.append("=" * len(header))

    report = "\n".join(lines)
    print(f"\n{report}")

    out_path = PROJECT_ROOT / "outputs" / "results" / "inference_timing.txt"
    out_path.write_text(report + "\n")
    print(f"\n[✓] Saved: {out_path}")


if __name__ == "__main__":
    main()
