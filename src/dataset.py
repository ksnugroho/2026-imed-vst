"""
dataset.py — iMED-VST
---------------------
Dataset loader untuk IMED "VIDEO MIXED EMOTION".

Struktur folder dataset:
  root/  (= "dataset/VIDEO MIXED EMOTION/")
    13_angry/
      Angry-Subject1/
        Gambar Angry-Subject1/   ← frame JPG
          Angry-Subject1 (1).jpg
          ...
    16_happy/
      Happy-Subject (1)/
        Gambar Happy-Subject1/
          ...
    ...

Split mode:
  - loso: leave-one-subject-out (fold = subject_id 1–15)
  - random: 80/20 per kelas
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# VideoTransform
# ---------------------------------------------------------------------------

class VideoTransform:
    """Terapkan augmentasi yang sama ke semua frame dalam satu clip."""

    def __init__(
        self,
        img_size: int = 224,
        is_train: bool = True,
        color_jitter: float = 0.4,
        random_grayscale: float = 0.1,
        flip_prob: float = 0.5,
        mean: tuple = (0.485, 0.456, 0.406),
        std:  tuple = (0.229, 0.224, 0.225),
    ):
        import torchvision.transforms as T
        import torchvision.transforms.functional as F

        self.is_train  = is_train
        self.flip_prob = flip_prob
        self.F = F

        if is_train:
            self.spatial = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.ColorJitter(color_jitter, color_jitter, color_jitter, 0.1),
                T.RandomGrayscale(p=random_grayscale),
            ])
        else:
            self.spatial = T.Compose([
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
            ])

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.is_train and random.random() < self.flip_prob:
            frames = [self.F.hflip(f) for f in frames]

        seed = random.randint(0, 2 ** 32 - 1)
        processed = []
        for frame in frames:
            random.seed(seed)
            torch.manual_seed(seed)
            frame = self.spatial(frame)
            frame = self.normalize(frame)   # [3, H, W]
            processed.append(frame)

        return torch.stack(processed, dim=0).permute(1, 0, 2, 3)  # [3, T, H, W]


# ---------------------------------------------------------------------------
# IMEDVideoDataset
# ---------------------------------------------------------------------------

class IMEDVideoDataset(Dataset):
    """
    IMED "VIDEO MIXED EMOTION" sebagai video FER dataset.

    Tiap sample = satu video clip (satu subjek, satu emosi) → T frame berurutan.
    Frame sudah diekstrak di folder "Gambar ..." tanpa perlu decode MP4.

    emotion_set:
      "basic" — 7 emosi murni: angry, disgust, fear, happy, neutral, sad, surprise
      "all"   — 19 kelas (7 basic + 12 compound)
    """

    BASIC_MAP = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "neutral": 4, "sad": 5, "surprise": 6,
    }

    _FOLDER_PREFIX = {
        "1":  "angrily-disgusted",  "2":  "angrily-surprised",
        "3":  "disgustedly-surprised", "4": "fear-angry",
        "5":  "fear-disgust",       "6":  "fear-surprise",
        "7":  "happy-disgust",      "8":  "happy-surprise",
        "9":  "sad-angry",          "10": "sad-disgust",
        "11": "sad-fear",           "12": "sad-surprise",
        "13": "angry",              "14": "disgust",
        "15": "fear",               "16": "happy",
        "17": "sad",                "18": "surprise",
        "19": "neutral",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",           # "train" | "val" | "test"
        split_mode: str = "loso",       # "loso" | "random"
        test_subject: int = 1,
        emotion_set: str = "basic",     # "basic" | "all"
        num_frames: int = 16,
        frame_interval: int = 4,
        train_ratio: float = 0.8,
        seed: int = 42,
        transform=None,
    ):
        self.root           = Path(root)
        self.num_frames     = num_frames
        self.frame_interval = frame_interval
        self.transform      = transform
        self.split          = split

        if emotion_set == "basic":
            self.class_to_idx = self.BASIC_MAP.copy()
        else:
            all_emotions = sorted(set(self._FOLDER_PREFIX.values()))
            self.class_to_idx = {e: i for i, e in enumerate(all_emotions)}

        self.classes = sorted(self.class_to_idx, key=lambda e: self.class_to_idx[e])

        all_clips = self._scan_clips(emotion_set)

        if split_mode == "loso":
            self.samples = self._split_loso(all_clips, test_subject, split)
        else:
            self.samples = self._split_random(all_clips, train_ratio, seed, split)

        print(f"[IMED/{split}] {len(self.samples)} clips | "
              f"{len(self.classes)} kelas | mode={split_mode}"
              + (f" | test_subj={test_subject}" if split_mode == "loso" else ""))

    # -- internal --

    def _scan_clips(self, emotion_set: str) -> list[dict]:
        clips = []
        for emotion_dir in sorted(self.root.iterdir()):
            if not emotion_dir.is_dir():
                continue
            prefix = emotion_dir.name.split("_")[0]
            emotion = self._FOLDER_PREFIX.get(prefix)
            if emotion is None:
                continue
            if emotion_set == "basic" and emotion not in self.BASIC_MAP:
                continue
            if emotion not in self.class_to_idx:
                continue
            label = self.class_to_idx[emotion]

            for subj_dir in sorted(emotion_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                subject_id = self._parse_subject_id(subj_dir.name)
                frame_folder = self._find_gambar_folder(subj_dir)
                if frame_folder is None:
                    continue
                frames = _sort_frames(list(frame_folder.glob("*.jpg")) + list(frame_folder.glob("*.png")))
                if len(frames) < self.num_frames:
                    continue
                clips.append({
                    "frame_folder": frame_folder,
                    "label":        label,
                    "subject_id":   subject_id,
                    "emotion":      emotion,
                })
        return clips

    @staticmethod
    def _find_gambar_folder(subj_dir: Path) -> Path | None:
        for d in subj_dir.iterdir():
            if d.is_dir() and d.name.lower().startswith("gambar"):
                return d
        return None

    @staticmethod
    def _parse_subject_id(folder_name: str) -> int:
        import re
        # Neutral: S1_T3_Neutral, S10_T3_Neutral, S3_T3_neutral
        m = re.match(r"[Ss](\d+)_", folder_name)
        if m:
            return int(m.group(1))
        # Standard: Angry-Subject1, Happy-Subject (1), angrily-disgusted-Subject1
        m = re.search(r"(\d+)\s*\)?$", folder_name)
        return int(m.group(1)) if m else 0

    def _split_random(self, clips: list, ratio: float, seed: int, split: str) -> list:
        """
        Stratified random split per class.
        ratio ≥ 0.75 → 2-way (train / val):  train=ratio, val=1-ratio
        ratio <  0.75 → 3-way (train / val / test):  train=ratio, val/test=(1-ratio)/2
        e.g. ratio=0.6 → 60 / 20 / 20.
        """
        from collections import defaultdict
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for c in clips:
            by_class[c["label"]].append(c)
        result = []
        three_way = ratio < 0.75
        for items in by_class.values():
            shuffled = items[:]
            rng.shuffle(shuffled)
            n_train = int(len(shuffled) * ratio)
            remainder = shuffled[n_train:]
            if not three_way:
                # Legacy 2-way: train gets [:n_train], val/test get all remainder
                if split == "train":
                    result.extend(shuffled[:n_train])
                else:
                    result.extend(remainder)
            else:
                # 3-way: val and test each get half the remainder
                n_val = max(0, len(remainder) // 2)
                if split == "train":
                    result.extend(shuffled[:n_train])
                elif split == "val":
                    result.extend(remainder[:n_val])
                else:  # "test"
                    result.extend(remainder[n_val:])
        return result

    def _split_loso(self, clips: list, test_subject: int, split: str) -> list:
        # Subjek diurutkan; val = subjek berikutnya secara sirkuler dari test.
        # Ini memastikan val dan test selalu berbeda (tidak ada leakage).
        subjects = sorted(set(c["subject_id"] for c in clips))
        idx      = subjects.index(test_subject)
        val_subject = subjects[(idx + 1) % len(subjects)]
        if split == "test":
            return [c for c in clips if c["subject_id"] == test_subject]
        if split == "val":
            return [c for c in clips if c["subject_id"] == val_subject]
        # train: semua kecuali test dan val
        return [c for c in clips if c["subject_id"] not in (test_subject, val_subject)]

    # -- Dataset interface --

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip = self.samples[idx]
        frames = _sort_frames(list(clip["frame_folder"].glob("*.jpg")) + list(clip["frame_folder"].glob("*.png")))
        indices = _temporal_sample(len(frames), self.num_frames, self.frame_interval)
        pil_frames = [Image.open(frames[i]).convert("RGB") for i in indices]
        tensor = self.transform(pil_frames) if self.transform else _to_tensor(pil_frames)
        return tensor, clip["label"]

    def class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes))
        for c in self.samples:
            counts[c["label"]] += 1
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * len(self.classes)


# ---------------------------------------------------------------------------
# Synthetic dataset (untuk sanity-check pipeline)
# ---------------------------------------------------------------------------

class SyntheticVideoDataset(Dataset):
    """Tensor acak tanpa file nyata — verifikasi pipeline saja."""

    def __init__(self, num_samples=200, num_classes=7, num_frames=16, img_size=224,
                 class_names=None):
        self.num_samples  = num_samples
        self.num_classes  = num_classes
        self.num_frames   = num_frames
        self.img_size     = img_size
        self.classes      = class_names or [str(i) for i in range(num_classes)]
        rng = random.Random(42)
        self.labels = [rng.randint(0, num_classes - 1) for _ in range(num_samples)]
        self.samples = [{"label": l} for l in self.labels]
        print(f"[Synthetic] {num_samples} dummy clips | {num_classes} kelas | T={num_frames}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clip = torch.randn(3, self.num_frames, self.img_size, self.img_size)
        return clip, self.labels[idx]

    def class_weights(self):
        counts = torch.zeros(self.num_classes)
        for l in self.labels:
            counts[l] += 1
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * self.num_classes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_frames(paths) -> list:
    """Sort frame paths numerically by the last integer in the filename stem.

    Handles both "(N).jpg" format and "_NN.jpg" format used in this dataset.
    Plain lexicographic sort would give wrong temporal order for (1),(10),(100),...
    """
    import re
    def key(p):
        nums = re.findall(r'\d+', Path(p).stem)
        return int(nums[-1]) if nums else 0
    return sorted(paths, key=key)


def _temporal_sample(total: int, num_frames: int, interval: int) -> list[int]:
    needed = num_frames * interval
    if total >= needed:
        start = random.randint(0, total - needed)
        return [start + i * interval for i in range(num_frames)]
    # clip pendek: sample uniform
    return [min(int(i * total / num_frames), total - 1) for i in range(num_frames)]


def _to_tensor(frames: list[Image.Image]) -> torch.Tensor:
    import torchvision.transforms.functional as F
    tensors = [F.to_tensor(f) for f in frames]
    return torch.stack(tensors, dim=0).permute(1, 0, 2, 3)


def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Sampler berbasis inverse class frequency untuk imbalanced FER."""
    cw = dataset.class_weights()
    sample_weights = torch.tensor([cw[s["label"]] for s in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
