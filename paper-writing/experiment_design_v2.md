# EXPERIMENT DESIGN V2 — iMED-VST
**Temporal Depth Enhancement untuk Video FER pada Dataset IMED**
W&B Project: `2026-imed-vst-v2`

---

## Perubahan dari V1

| Aspek | V1 (Deprecated) | V2 (Current) |
|---|---|---|
| Checkpoint selection | Val = Test subject (**leakage!**) | Val = subject berikutnya (circular), disjoint |
| Multi-seed | Tidak ada | 3 seeds: 42, 123, 456 — **semua** step |
| W&B project | `2026-imed-vst` | `2026-imed-vst-v2` |
| Step 4a multi-seed | Manual per-seed | Auto-derive seed-specific checkpoint dir |
| Encoder ablation | Transformer only | + GRU, + max pool (9 total variants) |
| Statistical test | Tidak ada | Wilcoxon signed-rank, effect size r |
| Identity leakage exp | Tidak ada | Random 60/20/20 vs LOSO |

> **PENTING:** Semua hasil V1 di `results_backup_pre_leakagefix/` TIDAK VALID
> karena checkpoint leakage. Semua step harus dijalankan ulang dengan V2.

---

## Dataset

- **Nama:** IMED "VIDEO MIXED EMOTION"
- **Jumlah clip:** 279 video clip
- **Subjek:** 15 subjek (S1–S15)
- **Kelas:** 19 total (7 basic + 12 compound), diatur via `emotion_set`
  - `basic`:  angry, disgust, fear, happy, neutral, sad, surprise
  - `all`:    19 kelas (termasuk angrily-disgusted, sad-angry, dll.)
- **Frame:** diekstrak ke JPG (folder "Gambar …")
- **Config T terpilih:** `num_frames=16`, `frame_interval=2` → window = 32 frame

---

## Arsitektur Model

### Backbone: Video Swin Transformer
| Backbone | Params | feat_dim | Pretrained |
|---|---|---|---|
| `swin3d_t` | 28M | 768 | Kinetics-400 |
| `swin3d_s` | 49M | 768 | Kinetics-400 |

**Forward pass:**
```
[B, 3, T, H, W]
  → patch_embed + pos_drop
  → Swin3D blocks (features)
  → LayerNorm (norm)
  → [B, D'=8, H'=7, W'=7, C=768]
```

### Temporal Depth Enhancement (TD) Module
Disisipkan setelah `norm`, menggantikan `avgpool`:

```
[B, D', H', W', C]
  → spatial avg pool H'×W' → [B, D', C]
  → encoder (lihat tabel) → [B, *, C]
  → temporal pool → [B, C]
  → head (LayerNorm → Dropout → Linear)
```

**encoder_type:**
| Nilai | Implementasi | Output |
|---|---|---|
| `"transformer"` | `TransformerEncoder` (Pre-LN, batch_first) | pool via mean/max/cls |
| `"gru"` | Bidirectional GRU; hidden=C//2×2dirs=C | concat last hidden states |
| `"none"` | Tidak ada encoder | langsung mean/max pool |

### Two-stage Training (Step 4a)
`freeze_backbone(True)` membekukan: `patch_embed`, `pos_drop`, `features`, `norm`.
TD module dan `head` tetap trainable di Stage 1.
- **Stage 1** (epoch 1 – `freeze_backbone_epochs`): hanya TD + head dioptimasi
- **Stage 2** (epoch `freeze_backbone_epochs+1` – akhir): seluruh model, backbone lr × 0.1

---

## Multi-seed Policy

Semua step menggunakan **3 seeds: 42, 123, 456** secara konsisten.

| Step | File | Seeds | Alasan |
|---|---|---|---|
| Step 0 | `run_ablation.py --seeds 42 123 456` | 3 | Konsistensi metodologi |
| Step 1 | `run_loso.py --seeds 42 123 456` | 3 | Robustness hasil |
| Step 2 | `run_loso.py --seeds 42 123 456` | 3 | Robustness hasil |
| Step 5 | `run_step5_td_ablation.py` | 1 (seed=42) | Search step; 9 variant × 15 fold = 135 run; 3 seeds = 405 run |
| Step 3 | `run_loso.py --seeds 42 123 456` | 3 | **Main result — wajib** |
| Step 4 | `run_loso.py --seeds 42 123 456` | 3 | **Main result — wajib** |
| Step 7 | `run_random_split_exp.py` | 3 (default) | Robust comparison |

**Pelaporan:** Mean ± Std across fold-averaged values per seed.
- Per seed → 15 nilai (per fold) → mean ± std across 15 folds
- Multi-seed summary → rata-rata metrik tiap fold di-aggregate across seeds

---

## Protokol Evaluasi

### LOSO Cross-Validation (Utama)
```
Untuk setiap fold f (1–15):
  test_subject   = subject f
  val_subject    = subject ((f mod 15) + 1)   [circular, disjoint dari test]
  train_subjects = semua 15 − {test, val}      = 13 subjek

Checkpoint selection : berdasarkan val_UAR (strict >, disimpan jika UAR > best sebelumnya)
Pelaporan            : test_WAR, test_UAR, test_MacroF1, test_WF1
                       (load best checkpoint dari val, eval di test)
```

> **Tidak ada overlap** antara val dan test subject di setiap fold.
> Ini menghasilkan 3-way split: 13 train / 1 val / 1 test.

### Metrik
| Metrik | Definisi | Alasan pilih |
|---|---|---|
| **WAR** | Weighted Average Recall = Accuracy | Standar perbandingan dengan prior work |
| **UAR** | Unweighted Average Recall = Macro-Recall | Checkpoint selection (robust terhadap class imbalance) |
| **Macro-F1** | Mean F1 per kelas (unweighted) | Balanced view precision+recall |
| **W-F1** | Weighted F1 | Informasi tambahan distribusi kelas |

---

## Urutan Eksperimen

```
Step 0  (temporal window search — random split, swin3d_t, backbone terkunci)
   ↓
Step 1  (backbone comparison: swin3d_t vs swin3d_s — LOSO, no-TD)
   ↓
Step 2  (loss function: CE vs Focal — LOSO, backbone terpilih)
   ↓
Step 5  (TD architecture ablation — single seed, search step)
   ↓
Step 3  (TD module terpilih, 7-class — kontribusi utama)
   ↓
Step 4a / 4b  (19 kelas: two-stage vs direct fine-tune)
   ↓
Step 6  (Wilcoxon statistical significance test)
Step 7  (Identity leakage experiment)
```

---

## Step 0 — Temporal Window Search

**Tujuan:** Cari kombinasi `num_frames` × `frame_interval` optimal.
**Backbone:** Dikunci ke `swin3d_t` untuk mengisolasi variabel temporal (backbone comparison di Step 1).
**Split:** Random 80/20 stratified per kelas (search step, bukan pelaporan final).
**Metrics dilaporkan:** val set (bukan test — tidak ada test set di Step 0).
**Multi-seed:** Ya (`--seeds 42 123 456`).

```bash
conda run -n ksn_cv_task python run_ablation.py --seeds 42 123 456
```

**Output:**
- `results/ablation_temporal_s42.csv`, `_s123.csv`, `_s456.csv`  — per seed, semua config
- `results/ablation_temporal_agg.csv`  — mean ± std per config across seeds

| Config | T | interval | Window | Backbone |
|---|---|---|---|---|
| A | 8 | 2 | 16 fr | swin3d_t |
| B | 8 | 4 | 32 fr | swin3d_t |
| C | 16 | 2 | 32 fr | swin3d_t |
| D | 16 | 4 | 64 fr | swin3d_t |
| E | 32 | 2 | 64 fr | swin3d_t |

**Konfigurasi terpilih (dari V1):** C — T=16, interval=2, window=32 fr.

---

## Step 1 — Backbone Selection

**Tujuan:** swin3d_t (28M) vs swin3d_s (49M).
**Config:** CE loss, no TD, 7-class, T=16 interval=2 (dari Step 0).

```bash
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_t --run_name step1_swin3d_t_7cls \
    --seeds 42 123 456

conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --run_name step1_swin3d_s_7cls \
    --seeds 42 123 456
```

**Output:**
- `results/step1_swin3d_{t,s}_7cls_s{42,123,456}.csv`
- `results/step1_swin3d_{t,s}_7cls_agg_summary.txt`

---

## Step 2 — Loss Function Ablation

**Tujuan:** CE vs Focal loss.
**Config:** Backbone terpilih dari Step 1, no TD, 7-class.

```bash
# CE — sudah dari Step 1 terbaik
# Focal:
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --loss_type focal \
    --run_name step2_focal_7cls --seeds 42 123 456
```

---

## Step 5 — TD Architecture Ablation

**Tujuan:** Justifikasi setiap hyperparameter TD module (menjawab reviewer: "why this specific config?").
**Multi-seed:** Single seed (42) — search step dengan 9 variant × 15 fold = 135 run.
**Config:** swin3d_s + CE + 7-class (identik dengan Step 1 terbaik agar comparable).

```bash
conda run -n ksn_cv_task python run_step5_td_ablation.py

# Atau hanya satu variant:
conda run -n ksn_cv_task python run_step5_td_ablation.py --variant td_cls

# Atau mulai dari variant tertentu:
conda run -n ksn_cv_task python run_step5_td_ablation.py --start_variant td_4x
```

| Variant | Encoder | L | PE | FFN | Pool | Pertanyaan dijawab |
|---|---|---|---|---|---|---|
| `td_base` | transformer | 1 | ✗ | 1536 | mean | Baseline TD (= Step 3) |
| `td_pe` | transformer | 1 | ✓ | 1536 | mean | Apakah PE membantu? |
| `td_2l` | transformer | 2 | ✗ | 1536 | mean | Lebih dalam = lebih baik? |
| `td_4x` | transformer | 1 | ✗ | 3072 | mean | FFN 4× (standar) membantu? |
| `td_cls` | transformer | 1 | ✗ | 1536 | cls | CLS token vs mean pool? |
| `td_full` | transformer | 2 | ✓ | 3072 | cls | Semua perbaikan kombinasi |
| `td_pe_cls` | transformer | 1 | ✓ | 1536 | cls | PE + CLS minimal |
| `td_maxpool` | none | — | — | — | max | Baseline temporal max pool |
| `td_gru` | gru | 1 | — | 1536 | — | Recurrent alternative |

**Output:**
- `results/step5_{variant}.csv` per variant
- `results/step5_td_ablation_comparison.txt` — tabel perbandingan semua variant

---

## Step 3 — TD Module (7-class, Kontribusi Utama)

**Tujuan:** Buktikan TD module meningkatkan performa vs baseline no-TD.
**Config:** Backbone + loss terpilih, `td_variant` = terbaik dari Step 5, multi-seed.

```bash
# Baseline no-TD: sudah dari Step 1/2 terbaik

# Dengan TD (td_gru terpilih dari Step 5):
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --use_td --td_variant td_gru \
    --run_name step3_td_gru_7cls --seeds 42 123 456
```

**Output:**
- `results/step3_td_gru_7cls_s{42,123,456}.csv`
- `results/step3_td_gru_7cls_agg_summary.txt`

---

## Step 4a — Two-Stage Training (19-class)

**Tujuan:** Transfer dari 7-class ke 19-class compound FER.
**Init checkpoint:** Otomatis mengambil seed-specific checkpoint dari Step 3.

```bash
conda run -n ksn_cv_task python run_loso.py \
    --emotion_set all --use_td --td_variant td_gru \
    --init_checkpoint_dir runs/step3_td_gru_7cls \
    --run_name step4a_twostage_19cls \
    --seeds 42 123 456
```

> **Catatan:** `--init_checkpoint_dir` diberikan **tanpa** suffix `_s{seed}`.
> Script secara otomatis akan mencari `runs/step3_td_gru_7cls_s42`,
> `runs/step3_td_gru_7cls_s123`, `runs/step3_td_gru_7cls_s456` untuk
> masing-masing seed. Jika tidak ditemukan, fallback ke path yang diberikan dengan warning.

**Dua-stage:**
- Stage 1 (epoch 1–`freeze_backbone_epochs`): backbone frozen (patch_embed + Swin blocks + norm); TD + head trainable
- Stage 2 (epoch `freeze_backbone_epochs+1` – akhir): backbone unfrozen, backbone lr × 0.1

## Step 4b — Direct Fine-tuning (19-class)

```bash
conda run -n ksn_cv_task python run_loso.py \
    --emotion_set all --use_td --td_variant td_gru \
    --run_name step4b_direct_td_gru_19cls --seeds 42 123 456
```

---

## Step 6 — Statistical Significance Testing

**Tujuan:** Menjawab reviewer: "apakah perbedaan statistik signifikan?"
**Metode:** Wilcoxon signed-rank test (two-sided), paired n=15 fold.
**Effect size:** r = Z/√N (r: 0.1=small, 0.3=medium, 0.5=large).

> **Catatan:** stat_significance.py membutuhkan per-fold CSV (bukan aggregated).
> Gunakan salah satu per-seed CSV (misal `_s42`). Jalankan untuk setiap seed
> dan periksa konsistensi signifikansi.

```bash
# Bandingkan no-TD vs TD (7-class) menggunakan seed 42
conda run -n ksn_cv_task python stat_significance.py \
    --a step1_swin3d_s_7cls_s42 --b step3_td_gru_7cls_s42

# Atau untuk semua pasangan TD variant (dari Step 5):
conda run -n ksn_cv_task python stat_significance.py --all_pairs
```

**CSV lookup logic** di `stat_significance.py`:
Mencari `results/step5_{name}.csv`, lalu `results/loso_{name}.csv`, lalu `results/{name}.csv`.

**Output per pasangan:** W-statistic, p-value (two-sided), effect size r, interpretasi.
**Threshold:** p < 0.05 → signifikan.

---

## Step 7 — Identity Leakage Experiment

**Tujuan:** Buktikan bahwa prior work's high scores disebabkan identity leakage (random split), bukan keunggulan arsitektur.

**Desain:**
| Kondisi | Split | Subjek |
|---|---|---|
| A (LOSO) | LOSO 13/1/1 | Train/val/test = disjoint subject |
| B (Random) | Random 60/20/20 | Subjek BISA overlap antar split |

Model, LR, epoch, augmentasi **identik** di kedua kondisi.

```bash
# Pakai hasil LOSO yang sudah ada dari Step 3 seed 42 (skip re-training kondisi A):
conda run -n ksn_cv_task python run_random_split_exp.py \
    --seeds 42 123 456 \
    --loso_csv outputs/results/step3_td_gru_7cls_s42.csv

# Tanpa loso_csv — training ulang kondisi A dari awal:
conda run -n ksn_cv_task python run_random_split_exp.py --seeds 42 123 456

# Hanya kondisi random:
conda run -n ksn_cv_task python run_random_split_exp.py \
    --condition random --seeds 42 123 456
```

> **Catatan:** `--loso_csv` harus menunjuk ke **per-seed CSV** (misal `step3_td_gru_7cls_s42.csv`),
> bukan ke file `_agg_summary.txt`. File `_agg.csv` tidak di-generate oleh kode;
> yang ada hanyalah `_agg_summary.txt` (plain text).

**Output:**
- `results/identity_leakage_random.csv` — 3 baris (kondisi B, per seed)
- `results/identity_leakage_comparison.txt` — tabel LOSO vs Random

---

## File dan Direktori

```
2026-imed-vst/
├── config.yaml
├── dataset.py                     # IMEDVideoDataset
│                                  #   _split_loso: 3-way (test/val circular/train)
│                                  #   _split_random: 2-way (ratio≥0.75) atau 3-way (ratio<0.75)
├── model.py                       # VideoSwinFER + TemporalEnhancement
│                                  #   feat_dim=768 (swin3d_t/s)
│                                  #   encoder_type: transformer | gru | none
│                                  #   freeze_backbone: bekukan patch_embed+features+norm
├── train.py                       # train_one_epoch, evaluate, set_seed, build_optimizer
├── metrics.py                     # WAR, UAR, MacroF1, WF1, per-class recall
├── run_ablation.py                # Step 0: temporal window search
│                                  #   --seeds → ablation_temporal_s{seed}.csv
├── run_loso.py                    # Steps 1–4: LOSO + multi-seed
│                                  #   --seeds 42 123 456
│                                  #   --init_checkpoint_dir auto-derives _s{seed}
│                                  #   TD_VARIANTS: 9 entries
├── run_step5_td_ablation.py       # Step 5: TD architecture ablation (single seed)
├── stat_significance.py           # Step 6: Wilcoxon signed-rank test
│                                  #   input: per-seed CSV (per-fold data wajib)
├── run_random_split_exp.py        # Step 7: Identity leakage experiment
│                                  #   --loso_csv: pakai per-seed CSV
├── EXPERIMENT_DESIGN_V2.md        # Dokumen ini
│
├── datasets/
│   └── VIDEO MIXED EMOTION/
│
├── results/
│   ├── ablation_temporal_s42.csv          # Step 0 per seed
│   ├── ablation_temporal_agg.csv          # Step 0 mean ± std across seeds
│   ├── step1_swin3d_t_7cls_s42.csv        # Step 1 per seed
│   ├── step1_swin3d_t_7cls_agg_summary.txt  # Step 1 aggregated (text, bukan CSV)
│   └── ...
│
└── runs/
    ├── ablation_A_s42/best.pt             # Step 0 checkpoints
    ├── step3_td_gru_7cls_s42/             # Step 3 fold checkpoints
    │   ├── fold_01_best.pt
    │   └── ...
    └── ...
```

> **Penting:** Multi-seed aggregate menyimpan ringkasan ke `*_agg_summary.txt` (plain text).
> Tidak ada `*_agg.csv` — hanya per-seed CSV yang berisi per-fold data.

---

## Perintah Lengkap (Urutan Eksekusi)

```bash
# Step 0 — temporal window (backbone dikunci swin3d_t, isolasi variabel temporal)
conda run -n ksn_cv_task python run_ablation.py --seeds 42 123 456

# Step 1 — backbone comparison
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_t --run_name step1_swin3d_t_7cls --seeds 42 123 456
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --run_name step1_swin3d_s_7cls --seeds 42 123 456

# Step 2 — loss function (jika perlu; sesuaikan backbone dengan hasil Step 1)
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --loss_type focal \
    --run_name step2_focal_7cls --seeds 42 123 456

# Step 5 — TD ablation (single seed; search step)
conda run -n ksn_cv_task python run_step5_td_ablation.py

# Step 3 — TD module (td_gru terpilih dari Step 5)
conda run -n ksn_cv_task python run_loso.py \
    --backbone swin3d_s --use_td --td_variant td_gru \
    --run_name step3_td_gru_7cls --seeds 42 123 456

# Step 4a — two-stage 19-class (auto-derive seed-specific init checkpoint)
conda run -n ksn_cv_task python run_loso.py \
    --emotion_set all --use_td --td_variant td_gru \
    --init_checkpoint_dir runs/step3_td_gru_7cls \
    --run_name step4a_twostage_19cls --seeds 42 123 456

# Step 4b — direct fine-tune 19-class (td_gru)
conda run -n ksn_cv_task python run_loso.py \
    --emotion_set all --use_td --td_variant td_gru \
    --run_name step4b_direct_td_gru_19cls --seeds 42 123 456

# Step 6 — statistical significance (pakai per-seed CSV)
conda run -n ksn_cv_task python stat_significance.py \
    --a step1_swin3d_s_7cls_s42 --b step3_td_gru_7cls_s42
# (ulangi untuk _s123 dan _s456 — periksa konsistensi)

# Step 7 — identity leakage
conda run -n ksn_cv_task python run_random_split_exp.py \
    --loso_csv outputs/results/step3_td_gru_7cls_s42.csv --seeds 42 123 456
```

---

## Ekspektasi Hasil (estimasi dari V1 — akan diverifikasi di V2)

| Step | WAR est. | UAR est. | Catatan |
|---|---|---|---|
| No-TD baseline (7cls) | ~35% | ~30% | Swin3D backbone only |
| TD-cls (7cls) | ~41% | ~35% | Dengan TD module |
| TD-cls (19cls, two-stage) | ~38% | ~30% | Compound FER |
| Random 60/20/20 (leaked) | ~70–80% | — | **Identity leakage inflated** |

> V2 mungkin menghasilkan angka berbeda karena evaluasi yang benar (leakage fixed).
> Selisih Random vs LOSO adalah argumen utama paper terhadap prior work.
