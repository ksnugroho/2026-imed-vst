"""
run_gradcam.py — Grad-CAM visualization untuk iMED-VST
-------------------------------------------------------
Menghitung Grad-CAM pada layer norm terakhir backbone (output [B, D', H', W', C])
dan menyimpan heatmap overlay di atas middle frame setiap clip.

Penggunaan:
  python run_gradcam.py --fold 4 --seed 42 --run_name step3_td_gru_7cls
  python run_gradcam.py --fold 4 --seed 42 --run_name step1_swin3d_s_7cls  # no-TD
  python run_gradcam.py --all_folds --seed 42 --run_name step3_td_gru_7cls

Output:
  results/gradcam/{run_name}_s{seed}_fold{fold:02d}/
    {emotion}_{clip_idx}_orig.jpg
    {emotion}_{clip_idx}_cam.jpg
  results/gradcam/{run_name}_s{seed}_fold{fold:02d}_grid.jpg  ← grid utama
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

warnings.filterwarnings("ignore")
BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from dataset import IMEDVideoDataset, VideoTransform, _sort_frames, _temporal_sample
from model import VideoSwinFER


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM pada model.norm (output [B, D', H', W', C]).

    Menggunakan register_full_backward_hook agar bekerja saat model.eval().
    """

    def __init__(self, model: VideoSwinFER):
        self.model      = model
        self._activation = None
        self._gradient   = None
        model.norm.register_forward_hook(self._fwd_hook)
        model.norm.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        # Simpan aktivasi; detach agar tidak ikut graph default
        self._activation = out.detach()          # [B, D', H', W', C]

    def _bwd_hook(self, module, grad_input, grad_output):
        # grad_output[0] = gradient mengalir keluar dari norm (ke arah input)
        self._gradient = grad_output[0].detach() # [B, D', H', W', C]

    def compute(self, inp: torch.Tensor, target_cls: int) -> np.ndarray:
        """Return normalized CAM [H', W'] ∈ [0, 1]."""
        # cuDNN RNN backward butuh training mode
        self.model.train()
        self.model.zero_grad()
        with torch.enable_grad():
            out   = self.model(inp)              # forward — hooks terpicu
            score = out[0, target_cls]
            score.backward()                     # backward — _bwd_hook terpicu
        self.model.eval()

        act  = self._activation[0]               # [D', H', W', C]
        grad = self._gradient[0]                 # [D', H', W', C]

        # Global-avg-pool gradient → channel weights [C]
        weights = grad.mean(dim=(0, 1, 2))       # [C]

        # Weighted sum over channel → [D', H', W']
        cam = (weights * act).sum(dim=-1)        # [D', H', W']

        # Average over temporal → [H', W']
        cam = cam.mean(dim=0).cpu()

        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam = cam / denom
        return cam.numpy()                       # [H', W']


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def cam_to_heatmap(cam: np.ndarray, pil_size: tuple[int, int]) -> Image.Image:
    """cam [H', W'] → PIL RGB heatmap resized to pil_size=(W, H)."""
    import matplotlib.cm as cm
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_size, Image.BILINEAR)
    ) / 255.0
    rgba = cm.jet(cam_resized)          # [H, W, 4]
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def overlay(orig: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    orig_arr = np.array(orig.convert("RGB")).astype(float)
    heat_arr = np.array(heat.convert("RGB")).astype(float)
    blended  = (1 - alpha) * orig_arr + alpha * heat_arr
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Frame loader (without augmentation — for visualization)
# ---------------------------------------------------------------------------

def load_clip_frames(clip: dict, num_frames: int, frame_interval: int,
                     img_size: int) -> tuple[torch.Tensor, list[Image.Image]]:
    """Return (tensor [1,3,T,H,W], list of PIL frames)."""
    frames = _sort_frames(
        list(clip["frame_folder"].glob("*.jpg")) +
        list(clip["frame_folder"].glob("*.png"))
    )
    indices    = _temporal_sample(len(frames), num_frames, frame_interval)
    pil_frames = [Image.open(frames[i]).convert("RGB") for i in indices]

    transform  = VideoTransform(img_size=img_size, is_train=False)
    tensor     = transform(pil_frames).unsqueeze(0)  # [1,3,T,H,W]
    return tensor, pil_frames


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(items: list[tuple[str, Image.Image, Image.Image]],
               cell_size: int = 224) -> Image.Image:
    """items = [(label, orig_pil, overlay_pil), ...]"""
    import math
    from PIL import ImageDraw, ImageFont

    n     = len(items)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    pad   = 4
    label_h = 20
    W = ncols * (cell_size * 2 + pad) + pad
    H = nrows * (cell_size + label_h + pad) + pad

    grid = Image.new("RGB", (W, H), (40, 40, 40))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, (label, orig, ov) in enumerate(items):
        row, col = divmod(i, ncols)
        x = pad + col * (cell_size * 2 + pad)
        y = pad + row * (cell_size + label_h + pad)

        orig_r = orig.resize((cell_size, cell_size))
        ov_r   = ov.resize((cell_size, cell_size))

        grid.paste(orig_r, (x, y + label_h))
        grid.paste(ov_r,   (x + cell_size, y + label_h))

        draw = ImageDraw.Draw(grid)
        draw.text((x + 2, y + 2), label, fill=(220, 220, 220), font=font)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fold(cfg, fold: int, run_name: str, seed: int, out_dir: Path,
             device: torch.device):
    ds_cfg = cfg["dataset"]
    num_frames    = ds_cfg["num_frames"]
    frame_interval= ds_cfg["frame_interval"]
    img_size      = ds_cfg.get("img_size", 224)
    class_names   = ds_cfg["class_names"]

    # Load test dataset (transform=None — frames loaded manually for visualization)
    test_ds = IMEDVideoDataset(
        root          = PROJECT_ROOT / ds_cfg["root"],
        split         = "test",
        split_mode    = "loso",
        test_subject  = fold,
        num_frames    = num_frames,
        frame_interval= frame_interval,
        emotion_set   = ds_cfg.get("emotion_set", "basic"),
    )

    if len(test_ds) == 0:
        print(f"  [!] No test clips for fold {fold}")
        return

    ckpt_path = PROJECT_ROOT / "outputs" / "runs" / f"{run_name}_s{seed}" / f"fold_{fold:02d}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "outputs" / "runs" / run_name / f"fold_{fold:02d}_best.pt"
    if not ckpt_path.exists():
        print(f"  [!] Checkpoint not found: {ckpt_path}")
        return

    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]   # backbone + td config tersimpan di checkpoint

    use_td    = ckpt_cfg["model"].get("use_td", False)
    td_config = ckpt_cfg["model"].get("td_config") if use_td else None

    model = VideoSwinFER(
        backbone    = ckpt_cfg["model"]["backbone"],
        num_classes = len(class_names),
        use_td      = use_td,
        td_config   = td_config,
    ).to(device)

    model.load_state_dict(state["model"])
    model.eval()

    gcam = GradCAM(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []   # (label, orig_pil, overlay_pil)

    for i, clip in enumerate(test_ds.samples):
        label_idx  = clip["label"]
        label_name = class_names[label_idx]

        tensor, pil_frames = load_clip_frames(clip, num_frames, frame_interval, img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred   = logits.argmax(dim=1).item()
            conf   = torch.softmax(logits, dim=1)[0, pred].item()

        # Use predicted class for CAM (what the model actually attends to)
        cam   = gcam.compute(tensor, pred)
        mid   = pil_frames[len(pil_frames) // 2]
        heat  = cam_to_heatmap(cam, mid.size)
        blend = overlay(mid, heat)

        correct = "✓" if pred == label_idx else "✗"
        title   = f"{label_name} → {class_names[pred]} {correct} ({conf:.0%})"
        items.append((title, mid, blend))

        # Save individual
        mid.save(out_dir / f"{label_name}_{i:02d}_orig.jpg")
        blend.save(out_dir / f"{label_name}_{i:02d}_cam.jpg")

    grid = build_grid(items, cell_size=224)
    grid_path = out_dir.parent / f"{out_dir.name}_grid.jpg"
    grid.save(grid_path, quality=92)
    print(f"  [✓] Grid: {grid_path}  ({len(items)} clips)")
    return grid_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold",      type=int, default=4,
                   help="Fold number (test subject). Default=4 (best fold WAR=100%)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--run_name",  type=str, default="step3_td_gru_7cls")
    p.add_argument("--all_folds", action="store_true",
                   help="Run Grad-CAM untuk semua 15 fold")
    args = p.parse_args()

    with open(BASE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override model config sesuai run_name
    if "td_gru" in args.run_name:
        cfg["model"]["use_td"]     = True
        cfg["model"]["td_variant"] = "td_gru"
        cfg["model"]["td_config"]  = {
            "encoder_type": "gru", "num_layers": 1,
            "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "mean",
        }
    elif "use_td" not in args.run_name:
        cfg["model"]["use_td"] = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | run_name: {args.run_name} | seed: {args.seed}")

    out_base = PROJECT_ROOT / "outputs" / "results" / "gradcam"
    folds = list(range(1, 16)) if args.all_folds else [args.fold]

    for fold in folds:
        out_dir  = out_base / f"{args.run_name}_s{args.seed}_fold{fold:02d}"
        print(f"\nFold {fold:2d} → {out_dir.name}")
        run_fold(cfg, fold, args.run_name, args.seed, out_dir, device)

    print("\n[✓] Selesai.")


if __name__ == "__main__":
    main()
