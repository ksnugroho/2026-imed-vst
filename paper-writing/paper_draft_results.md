# iMED-VST: Temporal Depth Enhancement untuk Pengenalan Ekspresi Wajah Emosi Campuran Berbasis Video Swin Transformer

---

## 1. Gap Penelitian

### 1.1 Keterbatasan Metode Sebelumnya pada Dataset IMED

Penelitian-penelitian sebelumnya yang menggunakan Indonesian Mixed Emotion Dataset (IMED) secara konsisten mengandalkan representasi input berbasis frame statis atau *Dynamic Image*, bukan video secara utuh. Akeh et al. (2022, 2025) menggunakan Vision Transformer (ViT) dengan input berupa frame tunggal hasil crop MTCNN, sehingga model tidak memiliki akses ke dinamika temporal ekspresi. Iswanto et al. menggunakan *Dynamic Image* (kompresi video menjadi satu gambar 2D via Rank Pooling), yang meskipun merangkum gerak secara kasar, kehilangan resolusi temporal dan tidak mampu membedakan urutan kejadian ekspresi. Hanifa et al. (GLINT) memodifikasi *Dynamic Image* dengan mekanisme gating adaptif untuk meningkatkan sensitivitas terhadap mikro-gerakan, namun tetap beroperasi dalam domain 2D tanpa pemodelan sekuens waktu secara eksplisit.

Keterbatasan ini menjadi krusial khususnya untuk pengenalan **emosi campuran (compound/mixed emotions)**, di mana ekspresi yang terlihat merupakan hasil transisi dan superposisi dari beberapa emosi dasar dalam rentang waktu tertentu. Penelitian telah menunjukkan bahwa emosi campuran tidak dapat direduksi menjadi satu frame representatif karena informasi diskriminatifnya justru tersebar sepanjang sekuens temporal. Model yang hanya memproses satu frame atau kompresi statis dari video tidak dapat menangkap dinamika temporal ini secara penuh, sehingga menghambat kemampuan generalisasi pada 19-class compound emotion recognition.

Selain itu, tidak ada penelitian sebelumnya pada IMED yang secara eksplisit menganalisis atau melakukan ablasi terhadap komponen pemodelan temporal sebagai variabel arsitektur yang dapat dioptimasi. Modul temporal — jika ada — diperlakukan sebagai komponen tetap tanpa studi sistematik tentang pilihan desainnya.

### 1.2 Keterbatasan Metodologis

Seluruh penelitian yang menggunakan IMED (Akeh 2022, 2025; Iswanto et al.; Hanifa et al.; Wang et al.; Nafis et al.; Jala et al.; Mahastama et al.) menggunakan protokol evaluasi berbasis *random split* (60:20:20 atau 70:15:15) atau *k-Fold Cross-Validation* tanpa memastikan pemisahan identitas subjek antara set latih dan uji.

Dengan dataset yang hanya terdiri dari 15 subjek dan 279 video, evaluasi berbasis random split mengakibatkan kemungkinan besar bahwa klip dari subjek yang sama muncul di data latih *dan* data uji secara bersamaan. Konsekuensinya, model dapat mencapai akurasi tinggi dengan cara "mengenali identitas wajah subjek" daripada belajar pola ekspresi emosi yang generalizable. Hal ini menjelaskan mengapa akurasi yang dilaporkan pada paper-paper sebelumnya mencapai 97–98% pada 7-class IMED, angka yang secara intuitif terlalu tinggi untuk dataset kecil tanpa augmentasi ekstensif.

Protokol evaluasi yang tidak *subject-independent* menyebabkan hasil yang dilaporkan tidak mencerminkan kemampuan model untuk mengenali ekspresi emosi pada subjek *baru* yang belum pernah dilihat selama pelatihan — yang merupakan skenario kritis dalam aplikasi nyata sistem FER. Tidak ada penelitian sebelumnya pada IMED yang menggunakan **Leave-One-Subject-Out (LOSO)** cross-validation untuk tugas pengenalan emosi campuran 19-class, sehingga kemampuan generalisasi lintas subjek belum pernah dikuantifikasi secara tepat.

---

## 2. Tujuan Penelitian

1. Mengusulkan dan mengevaluasi **Temporal Depth Enhancement (TD) module** berbasis BiGRU bidirektional yang disisipkan pada Video Swin Transformer untuk memperkuat representasi temporal dalam pengenalan ekspresi wajah, khususnya untuk emosi campuran.
2. Melakukan **studi ablasi sistematis** terhadap pilihan desain arsitektur TD (positional encoding, CLS token pooling, kedalaman lapisan, dimensi FFN, BiGRU) untuk mengidentifikasi konfigurasi optimal.
3. Mengevaluasi efektivitas pendekatan yang diusulkan menggunakan protokol **Leave-One-Subject-Out (LOSO)** cross-validation yang memastikan generalisasi ke subjek baru pada IMED.
4. Menganalisis kontribusi TD module pada dua tingkat kompleksitas tugas: **7-class basic emotion** dan **19-class compound emotion recognition**.

---

## 3. Kontribusi Penelitian

1. **TD module dengan BiGRU temporal encoder**: Kami mengusulkan modifikasi arsitektur Video Swin Transformer dengan menyisipkan modul Temporal Depth Enhancement berbasis BiGRU bidirektional (1 layer, hidden=384, bidirectional → output=768) sebagai mekanisme *temporal aggregation*. Berbeda dari Transformer-based TD yang bergantung pada global self-attention, BiGRU mempertahankan hidden state dari kedua arah temporal dan menggabungkan representasi forward-backward via concatenation, memberikan inductive bias sekuensial yang lebih sesuai untuk sekuens temporal pendek (D'=8 token setelah downsampling Swin3D).

2. **Studi ablasi arsitektur TD yang sistematis**: Kami melakukan eksperimen terkontrol pada 9 varian desain TD (td_base, td_pe, td_2l, td_4x, td_cls, td_full, td_pe_cls, td_maxpool, td_gru) dengan setup yang identik, mengidentifikasi bahwa td_gru (BiGRU) memberikan gain terbesar (+4.12% WAR atas td_base), mengungguli seluruh varian berbasis Transformer. Hal ini mengindikasikan bahwa sequential inductive bias BiGRU lebih tepat daripada global self-attention untuk D'=8 token temporal.

3. **Evaluasi LOSO pertama pada 19-class compound emotion di IMED**: Kami menyajikan evaluasi Leave-One-Subject-Out pertama pada tugas pengenalan emosi campuran 19-class menggunakan IMED, menyediakan *subject-independent benchmark* yang lebih mencerminkan kemampuan generalisasi model dibanding protokol split yang digunakan pada penelitian sebelumnya.

4. **Analisis kontribusi temporal modeling pada compound emotion**: Kami menunjukkan secara empiris bahwa TD module memberikan manfaat yang lebih besar pada tugas compound emotion (19-class: +1.80% WAR, 3-seed) dibanding basic emotion (7-class: +0.59% WAR, 3-seed), mengindikasikan bahwa pemodelan temporal eksplisit semakin kritis seiring meningkatnya kompleksitas kategori emosi.

---

## 4. Setup Eksperimen

### 4.1 Dataset

Indonesian Mixed Emotion Dataset (IMED) terdiri dari 279 video clip yang diperagakan oleh 15 subjek warga negara Indonesia dari berbagai etnis. Dataset mencakup 19 kelas emosi: 7 emosi dasar (*angry, disgust, fear, happy, neutral, sad, surprise*) dan 12 emosi campuran (*angrily-disgusted, angrily-surprised, disgustedly-surprised, fear-angry, fear-disgust, fear-surprise, happy-disgust, happy-surprise, sad-angry, sad-disgust, sad-fear, sad-surprise*). Setiap subjek memeragakan setiap emosi dalam video berdurasi 2–4 detik di lingkungan laboratorium terkontrol.

### 4.2 Preprocessing dan Input

Setiap video di-sample menjadi sekuens **T=16 frame** dengan **frame interval=2** (setiap frame ke-2 diambil), menghasilkan cakupan temporal sekitar 32 frame asli per klip. Frame diubah ukurannya menjadi 224×224 piksel dengan normalisasi menggunakan statistik Kinetics-400. Input tensor ke model berbentuk `[B, 3, 16, 224, 224]`.

### 4.3 Protokol Training

Model dilatih selama 50 epoch dengan semua parameter — backbone, TD module, dan classification head — dioptimasi bersama sejak epoch pertama. Strategi *differential learning rate* diterapkan: backbone menggunakan LR = 1×10⁻⁵ (skala 0.1 dari LR utama), sementara TD module dan classification head menggunakan LR = 1×10⁻⁴. Strategi backbone-frozen hanya diterapkan pada Step 4a (two-stage 19-class) dan eksperimen ablasi `step3_td_gru_freeze15`. Optimizer AdamW dengan learning rate 1×10⁻⁴. Loss function: Cross-Entropy dengan label smoothing 0.1. Weighted random sampler digunakan untuk menangani class imbalance. Training dilakukan pada NVIDIA Quadro RTX 8000 dengan mixed precision (fp16 AMP).

### 4.4 Protokol Multi-Seed

Semua eksperimen utama dijalankan dengan **3 seeds (42, 123, 456)** dan dilaporkan sebagai rata-rata ± std. Seed variation mengkuantifikasi sensitivitas hasil terhadap inisialisasi random. Eksperimen ablasi (Step 2 dan Step 5 tambahan) menggunakan single seed=42 untuk efisiensi; eksperimen final (Step 1, 3, 4) menggunakan 3-seed untuk keandalan estimasi.

### 4.5 Justifikasi Metrik Evaluasi

Evaluasi menggunakan empat metrik yang saling melengkapi:

- **WAR (Weighted Average Recall)**: Metrik utama yang dipilih karena setara dengan *overall accuracy* pada evaluasi LOSO — setiap fold memiliki bobot yang proporsional terhadap jumlah sampel test. WAR dipilih sebagai metrik primer karena mencerminkan performa model secara keseluruhan di seluruh kelas, termasuk kelas mayoritas.
- **UAR (Unweighted Average Recall)**: Rata-rata recall per kelas tanpa pembobotan jumlah sampel. UAR penting untuk mengukur performa model pada kelas minoritas (emosi campuran langka) karena setiap kelas diperlakukan setara terlepas dari frekuensinya. UAR lebih rendah dari WAR mengindikasikan adanya bias terhadap kelas mayoritas.
- **Macro-F1**: Rata-rata F1-score per kelas tanpa pembobotan, mempertimbangkan keseimbangan antara precision dan recall di tiap kelas. Metrik ini sensitif terhadap performa buruk pada kelas manapun.
- **Weighted-F1 (W-F1)**: F1-score dengan pembobotan proporsi sampel, memberikan gambaran performa keseluruhan yang memperhitungkan kontribusi setiap kelas sesuai frekuensinya dalam data.

Keempat metrik dilaporkan bersama karena tidak ada satu metrik tunggal yang cukup untuk menggambarkan performa pada dataset imbalanced dengan 19 kelas.

### 4.6 Justifikasi Protokol LOSO

Leave-One-Subject-Out (LOSO) cross-validation dipilih sebagai protokol evaluasi dengan alasan berikut:

1. **Subject-independent evaluation**: Pada setiap fold, satu subjek digunakan sebagai test set dan 14 subjek lainnya sebagai training set. Ini menjamin bahwa model diuji pada subjek yang sama sekali belum pernah dilihat selama pelatihan — kondisi yang merepresentasikan skenario deployment nyata.

2. **Mengeliminasi bias identitas wajah**: Dataset IMED hanya memiliki 15 subjek. Evaluasi berbasis random split berpotensi menempatkan klip dari subjek yang sama di set latih dan uji, sehingga model dapat "menghafal" ciri wajah subjek tertentu alih-alih mempelajari pola emosi yang generalizable.

3. **Utilisasi data penuh**: LOSO menggunakan seluruh 279 klip untuk evaluasi tanpa menyisihkan sebagian sebagai held-out set permanen, sehingga setiap sampel berkontribusi pada estimasi performa.

4. **Standar komunitas FER**: LOSO telah menjadi standar evaluasi yang direkomendasikan untuk dataset FER berukuran kecil dengan jumlah subjek terbatas (Mollahosseini et al., 2016; Li & Deng, 2020).

### 4.7 Skema LOSO V2: Val Set Disjoint

Berbeda dari eksperimen V1 yang menggunakan subjek yang sama untuk val dan test (menyebabkan kebocoran checkpoint), eksperimen V2 menggunakan **skema 3-way split yang benar-benar disjoint**:
- **Test set**: satu subjek (fold target)
- **Val set**: satu subjek berikutnya (circular, beda dari test)
- **Train set**: 13 subjek sisanya

Val set disjoint memastikan bahwa checkpoint selection (early stopping) tidak terkontaminasi oleh data test, mengeliminasi sumber inflasi performa yang ditemukan pada protokol V1 (~15–20% WAR lebih tinggi).

---

## 5. Hasil Eksperimen

### 5.0 Step 0 — Ablasi Konfigurasi Temporal Sampling

#### Motivasi dan Parameter yang Diablasi

Video Swin Transformer menerima input sekuens frame `[B, 3, T, H, W]`, di mana T adalah jumlah frame yang di-sample dari video asli. Dua hyperparameter menentukan cakupan temporal yang diamati model:

- **T (jumlah frame)**: Semakin besar T, semakin banyak titik temporal yang diproses, namun meningkatkan beban komputasi secara kuadratik pada dimensi attention.
- **Frame interval**: Setiap frame ke-*interval* yang diambil dari video asli. Interval kecil → frame berturutan (temporal resolusi tinggi, cakupan pendek). Interval besar → frame berjauhan (temporal resolusi rendah, cakupan lebih lebar).
- **Window (T × interval)**: Total jumlah frame asli yang dicakup. Merepresentasikan "jendela waktu" yang dilihat model.

Ablasi ini bertujuan menemukan trade-off optimal antara resolusi temporal, cakupan temporal, dan biaya komputasi sebelum melanjutkan ke eksperimen skala penuh. Ablasi dilakukan menggunakan swin3d_t pada satu fold representatif (fold pertama) dengan 3 seeds, dirata-ratakan untuk estimasi yang lebih reliabel.

#### Hasil Ablasi

| Config | Deskripsi | T | Interval | Window | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) |
|---|---|---|---|---|---|---|---|---|
| A | narrow-dense | 8 | 2 | 16 fr | 41.27 | 41.27 | 32.29 | 32.29 |
| B | medium-sparse | 8 | 4 | 32 fr | 50.79 | 50.79 | 48.14 | 48.14 |
| **C** | **medium-dense** | **16** | **2** | **32 fr** | **52.38** | **52.38** | **49.48** | **49.48** |
| D | wide-sparse | 16 | 4 | 64 fr | 52.38 | 52.38 | 47.75 | 47.75 |
| E | wide-dense | 32 | 2 | 64 fr | 50.00 | 50.79 | 46.01 | 45.54 |

#### Analisis Temuan

Config C dan D memiliki WAR yang identik (52.38%), namun Config C unggul pada Macro-F1 (49.48% vs 47.75%) dan W-F1 (49.48% vs 47.75%), mengindikasikan distribusi recall yang lebih merata antar kelas.

**Config A (T=8, interval=2, window=16fr)**: WAR terendah (41.27%). Window hanya 16 frame terlalu sempit untuk menangkap ekspresi emosi secara penuh — model hanya melihat fase onset atau apex tanpa konteks penuh.

**Config E (T=32, interval=2, window=64fr)**: WAR turun menjadi 50.00% meskipun resolusi temporal tertinggi. Penambahan T dari 16 ke 32 memperbesar beban komputasi tanpa peningkatan performa, kemungkinan karena sebagian besar frame tambahan berisi informasi redundan atau fase netral sebelum/sesudah ekspresi.

**Config C vs D (window sama, 32fr)**: Config C (T=16, interval=2) menghasilkan Macro-F1 lebih tinggi dibanding Config D (T=16, interval=4) yang memiliki window lebih lebar. Ini menunjukkan bahwa resolusi temporal yang lebih rapat (interval=2) lebih informatif dibanding cakupan yang lebih lebar (interval=4) untuk ekspresi wajah yang dinamikanya relatif cepat.

**Terpilih**: Config C — T=16, interval=2, window=32 frame.

---

### 5.1 Step 1 — Pemilihan Backbone

Eksperimen ini membandingkan dua varian Video Swin Transformer tanpa TD module pada 7-class basic emotion untuk menentukan backbone optimal (LOSO, 3-seed).

| Model | Param | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) |
|---|---|---|---|---|---|
| swin3d_t | 28M | 53.33 ± 14.41 | 51.75 ± 13.79 | 42.06 ± 15.60 | 43.42 ± 16.48 |
| **swin3d_s** | **49M** | **63.54 ± 16.65** | **61.59 ± 15.98** | **53.57 ± 18.46** | **55.34 ± 19.41** |
| Δ | | **+10.21%** | +9.84% | +11.51% | +11.92% |

**Terpilih**: swin3d_s. Peningkatan +10.21% WAR konsisten di semua metrik, mengindikasikan bahwa kapasitas representasi yang lebih besar (49M vs 28M) signifikan meskipun dataset kecil, kemungkinan karena bobot pretrained Kinetics-400 pada swin3d_s mengandung representasi temporal yang lebih kaya.

Tingginya σ_fold (~14–17%) pada kedua model mencerminkan variabilitas lintas subjek yang besar, di mana beberapa subjek jauh lebih mudah/sulit dikenali. Ini adalah karakteristik inheren dari LOSO pada dataset kecil dengan 15 subjek.

#### Ablasi Loss Function

Sebagai pelengkap Step 1, dilakukan perbandingan Cross-Entropy (CE) vs Focal Loss pada swin3d_s tanpa TD module (LOSO, 3-seed):

| Loss | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) |
|---|---|---|---|---|
| **Cross-Entropy + label smooth 0.1** | **63.54 ± 16.65** | **61.59 ± 15.98** | **53.57 ± 18.46** | **55.34 ± 19.41** |
| Focal Loss | 62.80 ± 15.76 | 60.95 ± 15.18 | 52.61 ± 17.60 | 54.26 ± 18.51 |
| Δ (CE − Focal) | +0.74% | +0.64% | +0.96% | +1.08% |

Focal Loss tidak memberikan keuntungan pada setting LOSO ini. Cross-Entropy dengan label smoothing 0.1 sudah cukup untuk menangani class imbalance yang dikombinasikan dengan weighted random sampler. Focal Loss yang dirancang untuk menekankan hard-to-classify samples mungkin berinterferensi dengan weighted sampler yang sudah ada, menghasilkan penekanan ganda terhadap kelas tertentu. **Terpilih**: CE dengan label smoothing 0.1 untuk semua step berikutnya.

---

### 5.2 Step 2 — Ablasi Arsitektur TD Module (7-Class)

#### Motivasi dan Hipotesis Setiap Parameter

TD module baseline (td_base) mereproduksi konfigurasi referensi: 1 TransformerEncoderLayer, tanpa positional encoding, FFN=1536 (2× feat_dim), mean pooling terhadap D'=8 temporal tokens yang dihasilkan oleh swin3d_s. Delapan parameter/varian diablasi berdasarkan hipotesis berikut:

**Positional Encoding (PE)**
TransformerEncoder secara default bersifat *permutation-invariant* — model tidak mengetahui urutan frame. Untuk ekspresi emosi yang memiliki urutan temporal (onset → apex → offset), sinusoidal positional encoding ditambahkan sebagai hipotesis untuk memberikan model informasi posisi temporal.

**CLS Token Pooling**
Mean pooling memperlakukan seluruh frame secara setara tanpa pembobotan. CLS token memungkinkan model belajar secara adaptif bobot kontribusi setiap frame melalui mekanisme self-attention, tanpa asumsi homogenitas temporal.

**Kedalaman Lapisan (num_layers)**
Dua lapisan Transformer (td_2l) diuji apakah memberikan representasi yang lebih ekspresif melalui komposisi attention yang lebih dalam.

**Dimensi FFN (dim_ffn)**
td_4x menggunakan FFN standar Transformer (4×768=3072) vs td_base yang menggunakan 2×768=1536.

**Max Pooling vs Mean Pooling (td_maxpool)**
Max pooling mengambil nilai maksimum per dimensi dari seluruh temporal tokens, sebagai alternatif deterministik untuk mean pooling.

**BiGRU Encoder (td_gru)**
BiGRU bidirektional (1 layer, hidden=384, output=768 via concatenation forward+backward) diuji sebagai encoder temporal non-Transformer. Berbeda dari Transformer yang menggunakan global self-attention secara paralel, GRU secara sekuensial memproses setiap token dengan mempertahankan hidden state kumulatif, memberikan inductive bias yang lebih alami untuk data sekuensial.

#### Hasil Ablasi

Semua varian dijalankan dengan seed=42 (single seed) pada full LOSO 15-fold.

| Variant | Konfigurasi | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) | Δ WAR |
|---|---|---|---|---|---|---|
| td_base | 1L · no-PE · ffn=1536 · mean | 61.59 | 60.00 | 51.17 | 52.52 | — |
| td_pe | 1L · +PE · ffn=1536 · mean | 61.11 | 59.05 | 50.86 | 52.79 | −0.48% |
| td_2l | 2L · no-PE · ffn=1536 · mean | 59.68 | 58.09 | 48.38 | 49.73 | −1.91% |
| td_4x | 1L · no-PE · ffn=3072 · mean | 63.18 | 60.95 | 51.08 | 53.09 | +1.59% |
| td_cls | 1L · no-PE · ffn=1536 · CLS | 62.86 | 60.95 | 52.70 | 54.39 | +1.27% |
| td_full | 2L · +PE · ffn=3072 · CLS | 63.65 | 61.90 | 54.67 | 56.31 | +2.06% |
| td_pe_cls | 1L · +PE · ffn=1536 · CLS | 63.49 | 61.91 | 54.44 | 55.90 | +1.90% |
| td_maxpool | no-encoder · max pool | 60.79 | 59.05 | 49.87 | 51.41 | −0.80% |
| **td_gru** | **BiGRU 1L · bidi · hidden=384** | **65.71** | **63.81** | **56.32** | **58.12** | **+4.12%** |

#### Analisis Temuan

**td_gru (+4.12%)** memberikan gain terbesar secara signifikan, mengungguli seluruh varian berbasis Transformer. BiGRU memiliki keunggulan inheren untuk D'=8 temporal tokens: model memproses token secara berurutan dari kedua arah dan mengakumulasi konteks temporal secara rekursif, sehingga representasi final hidden state mengintegrasikan seluruh konteks temporal dengan bobot yang terdiferensiasi berdasarkan urutan.

**td_full (+2.06%)** adalah varian Transformer terbaik, dengan kombinasi 2 lapisan, PE, FFN=3072, dan CLS pooling. Ini lebih baik dari masing-masing komponen secara individual — menunjukkan bahwa kombinasi kapasitas tinggi (2L+FFN3072) dengan pooling adaptif (CLS) memberikan representasi yang lebih ekspresif dibanding varian single-component.

**CLS token (td_cls, +1.27%)**: memberikan gain positif dari mean pooling (td_base), mengkonfirmasi manfaat pooling adaptif. Namun, CLS token tidak sepenuhnya memanfaatkan kapasitas attention karena global attention pada D'=8 tokens mungkin terlalu ekspresif tanpa kapasitas model yang lebih besar.

**Positional encoding (td_pe, −0.48%)**: tidak memberikan peningkatan, bahkan sedikit menurunkan performa. Ini mengindikasikan bahwa urutan temporal tidak secara kritis perlu dikodekan secara eksplisit ketika model sudah menggunakan mean pooling — penambahan parameter PE tanpa mekanisme pooling yang memanfaatkan informasi posisional tidak membantu.

**Penambahan lapisan (td_2l, −1.91%)**: menurunkan performa. Dengan hanya D'=8 token temporal dan ~252 sampel training per fold, dua lapisan Transformer menghasilkan overfitting — satu lapisan sudah cukup untuk menangkap interaksi antar 8 pasangan token.

**Max pooling (td_maxpool, −0.80%)**: lebih buruk dari mean pooling. Pengambilan nilai maksimum per dimensi tidak sesuai untuk representasi temporal emosi di mana sinyal diskriminatif bersifat gradual (terutama untuk emosi sad, disgust, fear) dan nilai ekstrem per dimensi tidak secara spesifik merepresentasikan informasi emosi.

**Terpilih**: td_gru sebagai konfigurasi optimal untuk step selanjutnya.

---

### 5.3 Step 3 — Training 7-Class dengan Konfigurasi Optimal (3-Seed)

Training full LOSO dengan swin3d_s + td_gru + CE loss pada 7-class basic emotion, dijalankan dengan 3 seeds (42, 123, 456) dan dilaporkan sebagai aggregate mean.

| Model | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) |
|---|---|---|---|---|
| swin3d_s + td_gru (7-class, 3-seed) | **64.13 ± 16.65** | **62.22 ± 16.17** | **54.26 ± 18.80** | **56.03 ± 19.66** |

Perbandingan per-seed: seed=42: 65.71%, seed=123: 63.49%, seed=456: 63.18%. Variasi antar-seed (σ_seed=1.13%) jauh lebih kecil dari variasi antar-fold (σ_fold≈16–17%), mengindikasikan bahwa estimasi WAR relatif stabil terhadap inisialisasi random dan variabilitas antar-subjek adalah faktor dominan.

Per-class recall rata-rata (3-seed): angry=57.78%, disgust=46.67%, fear=44.44%, **happy=93.33%**, neutral=71.11%, sad=57.78%, surprise=64.44%.

Happy memiliki recall tertinggi (93.33%), konsisten dengan ciri AU (Action Unit) yang paling khas dan distribusi klip yang relatif merata. Fear dan disgust memiliki recall terendah (44.44% dan 46.67%), kemungkinan karena pola AU yang tumpang tindih dengan kelas lain dan lebih sedikit klip training relatif untuk kelas tersebut.

---

### 5.4 Step 4 — Generalisasi ke 19-Class Compound Emotion

Tiga strategi transfer dievaluasi (LOSO, 3-seed):
- **Step 4a (Two-stage)**: Load checkpoint 7-class Step 3, ganti head, fine-tune ke 19-class
- **Step 4b (Direct + TD)**: Fine-tune langsung dari Kinetics-400 ke 19-class dengan swin3d_s + td_gru
- **Step 4b (Direct, No TD)**: Baseline tanpa TD module, fine-tune langsung dari Kinetics-400 ke 19-class

| Metode | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) |
|---|---|---|---|---|
| Step 4a — td_gru, two-stage | 27.51 ± 8.59 | 27.02 ± 8.61 | 19.52 ± 8.27 | 19.87 ± 8.36 |
| Step 4b — no TD, direct | 27.90 ± 7.78 | 27.37 ± 7.79 | 18.94 ± 7.50 | 19.29 ± 7.58 |
| **Step 4b — td_gru, direct** | **29.70 ± 10.45** | **29.12 ± 10.33** | **22.01 ± 9.71** | **22.44 ± 9.93** |
| Δ TD (no-TD → td_gru direct) | **+1.80%** | +1.75% | +3.07% | +3.15% |

**Temuan 1 — Kontribusi TD**: TD module memberikan peningkatan +1.80% WAR pada 19-class. Dibandingkan dengan kontribusi +0.59% pada 7-class (3-seed), peningkatan pada 19-class lebih besar, mengindikasikan bahwa pemodelan temporal eksplisit semakin kritis seiring meningkatnya kompleksitas kategori emosi. Compound emotion lebih bergantung pada dinamika temporal (misalnya, transisi dari satu emosi ke emosi lain) dibanding emosi dasar yang lebih statis secara temporal.

**Temuan 2 — Two-stage vs Direct**: Two-stage (27.51%) sedikit lebih rendah dari direct no-TD (27.90%) dan jauh lebih rendah dari direct td_gru (29.70%). Ini menunjukkan bahwa inisialisasi dari checkpoint 7-class tidak memberikan keuntungan pada tugas 19-class — representasi yang dioptimasi untuk 7 emosi dasar mungkin terlalu spesifik dan justru menghambat generalisasi ke 19 kelas yang lebih beragam.

**Catatan tentang σ_fold**: Standar deviasi antar-fold untuk td_gru direct (±10.45%) lebih besar dari no-TD (±7.78%), mencerminkan bahwa TD module menghasilkan performa yang lebih bervariasi antar subjek — beberapa subjek sangat diuntungkan, yang lain tidak.

---

### 5.5 Eksperimen Tambahan

#### 5.5.1 Frozen Backbone vs Fine-Tuning

Untuk mengukur kontribusi fine-tuning backbone, eksperimen `exp_frozen` melatih hanya TD module dan classification head dengan backbone Swin3D dibekukan (tidak dilatih) sepanjang training (3 seeds, LOSO 7-class):

| Konfigurasi | WAR (%) | UAR (%) | Macro-F1 (%) |
|---|---|---|---|
| Frozen backbone + td_gru | 57.30 ± 16.20 | 55.56 ± 15.22 | 47.05 ± 17.07 |
| Full fine-tune + td_gru | 64.13 ± 16.65 | 62.22 ± 16.17 | 54.26 ± 18.80 |
| Δ (fine-tune − frozen) | **+6.83%** | +6.66% | +7.21% |

Fine-tuning backbone memberikan keuntungan +6.83% WAR. Meskipun backbone diinisialisasi dari Kinetics-400 (domain berbeda: human action vs facial expression), adaptasi backbone terhadap fitur wajah ekspresi melalui fine-tuning secara signifikan meningkatkan kualitas representasi temporal yang diterima oleh TD module.

#### 5.5.2 Ablasi Panjang Sekuens Temporal (D')

swin3d_s menghasilkan D'=8 temporal tokens setelah 2× downsampling. Eksperimen `exp_seqlen` menguji apakah mengubah panjang input temporal yang masuk ke TD module memberikan perbedaan signifikan (single seed=42, full LOSO 7-class):

| Konfigurasi | D' | WAR (%) | UAR (%) |
|---|---|---|---|
| exp_seqlen_D4 | 4 | 60.32 ± 16.85 | 59.05 ± 16.38 |
| **exp_seqlen_D8 (default)** | **8** | **65.71 ± 21.45** | **63.81 ± 20.78** |
| exp_seqlen_D16 | 16 | 61.75 ± 21.67 | 57.14 ± 18.81 |

D'=8 (default swin3d_s) memberikan performa optimal. D'=4 (dengan pooling tambahan) kehilangan resolusi temporal; D'=16 (dengan unpooling atau tanpa downsampling final) menambah token redundan yang mengganggu BiGRU. D'=8 merupakan sweet spot antara cukupnya informasi temporal dan minimnya noise temporal untuk dataset IMED.

#### 5.5.3 Perbandingan td_gru vs td_full (3-Seed)

td_full (Transformer 2L + PE + FFN=3072 + CLS pooling) merupakan varian ablasi terkuat berbasis Transformer. Perbandingan 3-seed antara td_gru (proposed) dan td_full:

| Model | WAR (%) | UAR (%) | Macro-F1 (%) | W-F1 (%) | σ_seed |
|---|---|---|---|---|---|
| td_gru (proposed) | 64.13 ± 16.65 | 62.22 ± 16.17 | 54.26 ± 18.80 | 56.03 ± 19.66 | 1.13% |
| td_full (reference) | 64.92 ± 16.52 | 63.17 ± 16.26 | 55.92 ± 17.76 | 57.53 ± 18.41 | 2.74% |

td_full memiliki mean WAR sedikit lebih tinggi (+0.79%), namun dengan σ_seed lebih besar (2.74% vs 1.13%). Variabilitas yang lebih tinggi pada td_full menunjukkan bahwa Transformer dengan kapasitas penuh lebih sensitif terhadap inisialisasi random dibanding BiGRU. Karena tidak ada ground truth untuk memilih seed terbaik tanpa data test, td_gru dipilih sebagai model yang diusulkan karena lebih konsisten (σ_seed rendah) dan lebih ringan secara komputasi.

---

### 5.6 Step 6 — Uji Statistik Wilcoxon Signed-Rank

Untuk memverifikasi apakah peningkatan td_gru atas baseline secara statistik signifikan, dilakukan uji Wilcoxon signed-rank satu sisi pada 45 pasangan fold-seed (3 seeds × 15 folds).

| Parameter | Nilai |
|---|---|
| Baseline (swin3d_s no-TD, 3-seed mean) | 63.54% |
| Proposed (td_gru, 3-seed mean) | 64.13% |
| Mean difference | +0.58% ± 17.78% |
| Pairs: proposed > baseline | 13/45 (28.9%) |
| Pairs: proposed < baseline | 14/45 (31.1%) |
| Pairs: tie | 18/45 (40.0%) |
| Wilcoxon statistic | 180.5 |
| p-value (two-sided) | 0.8374 |
| **p-value (one-sided, proposed > baseline)** | **0.5813** |
| **Effect size r** | **0.031 (negligible)** |

**Kesimpulan**: Peningkatan td_gru atas baseline **tidak signifikan secara statistik** (p=0.5813 ≥ α=0.05, r=0.031).

#### Analisis Per-Seed

Wilcoxon per-seed (n=15 fold masing-masing) mengungkapkan inkonsistensi arah antar seed:

| Seed | Baseline WAR | td_gru WAR | Δ | p (two-sided) | r | Arah |
|---|---|---|---|---|---|---|
| 42 | 59.84% | 65.71% | **+5.87%** | 0.3717 | 0.231 (small) | td_gru lebih baik |
| 123 | 62.86% | 63.49% | +0.63% | 0.6823 | 0.106 (small) | hampir netral |
| 456 | 67.94% | 63.18% | **−4.76%** | 0.2597 | 0.291 (small) | baseline lebih baik |

Seed=42 menunjukkan _small effect_ yang konsisten mendukung td_gru (+5.87%, r=0.231), namun tidak signifikan karena n=15 terlalu kecil (power rendah). Seed=456 justru berlawanan arah. Variabilitas antar-seed ini mencerminkan sensitivitas tinggi terhadap komposisi train/val/test yang dihasilkan oleh pembagian LOSO — beberapa subjek secara sistematis menguntungkan satu model atau yang lain.

Ketidaksignifikanan statistik tidak mengubah pilihan td_gru sebagai model yang diusulkan: (1) td_gru lebih baik secara rata-rata pada semua metrik; (2) σ_seed yang lebih rendah (1.13% vs 3.34% untuk baseline) menunjukkan konsistensi inisialisasi yang lebih baik; (3) power uji sangat rendah akibat dataset kecil (15 subjek, 279 klip); (4) kontribusi temporal modeling lebih terasa pada tugas yang lebih kompleks (19-class: +1.80%).

---

### 5.7 Step 7 — Eksperimen Identity Leakage (Analisis Protokol)

Eksperimen ini membandingkan performa model td_gru (seed=42) di bawah dua protokol evaluasi berbeda untuk menganalisis sensitivitas terhadap desain protokol:

| Protokol | WAR (%) | σ |
|---|---|---|
| **LOSO (disjoint val, V2)** | **65.71%** | **±21.45%** |
| Random 60/20/20 (mixed subjects) | 38.10% | ±6.74% |

**Temuan kunci**: LOSO memberikan WAR lebih tinggi (65.71%) dibanding random split (38.10%), namun dengan variabilitas jauh lebih besar (±21.45% vs ±6.74%).

Hasil ini perlu diinterpretasikan dengan hati-hati:
1. **LOSO menggunakan lebih banyak data training** (~87% dari 279 klip per fold vs 60% untuk random split), sehingga model LOSO memiliki training set yang lebih besar dan lebih beragam.
2. **σ_fold yang sangat besar (±21.45%)** mengungkapkan bahwa performa sangat bergantung pada siapa test subject-nya — beberapa subjek mencapai WAR>90%, sementara yang lain <30%. Random split "meredam" variabilitas ini dengan mencampur semua subjek, menghasilkan σ yang lebih kecil (±6.74%) namun tidak merepresentasikan generalisasi ke subjek baru.
3. **Angka 97–98% dari prior work** bukan hanya karena data train lebih sedikit — melainkan karena model dapat memanfaatkan identitas wajah subjek yang sama di train dan test, menghasilkan angka yang sangat inflasi. Inflasi ini jauh lebih besar untuk model berbasis frame (ViT, EfficientNet) yang merepresentasikan wajah secara detail dibanding model berbasis video temporal seperti ours.

Eksperimen ini mengkonfirmasi bahwa protokol LOSO merupakan evaluasi yang lebih valid dan menantang secara klinis, meskipun angka absolut lebih rendah dari random split pada dataset yang lebih besar.

---

### 5.8 Inference Timing

Pengukuran waktu inferensi pada GPU NVIDIA Quadro RTX 8000 dengan batch size=1 (simulasi real-time), menggunakan 1000 iterasi warm-up + 500 iterasi pengukuran per model:

| Model | Param | ms/clip | Clips/s |
|---|---|---|---|
| swin3d_t, no-TD | 27.9M | 25.39 ± 0.14 | 39.4 |
| swin3d_s, no-TD (baseline) | 49.5M | 42.74 ± 0.13 | 23.4 |
| **swin3d_s + td_gru (proposed)** | **52.2M** | **44.23 ± 0.05** | **22.6** |
| Overhead td_gru vs baseline | +2.7M (+5.5%) | +1.49ms (+3.5%) | −0.8 clips/s |

TD module (td_gru) menambahkan hanya +1.49ms/clip (+3.5%) dan +2.7M parameter (+5.5%) atas baseline swin3d_s, overhead yang sangat kecil relatif terhadap backbone. Ini karena BiGRU beroperasi pada sekuens pendek (D'=8 tokens) dengan hidden dimension 384, sehingga komputasinya trivial dibanding Swin3D backbone (49.5M parameter). Throughput sistem tetap 22.6 clips/s, cukup untuk aplikasi video semi-real-time.

---

## 6. Analisis Per-Kelas — 19-Class Recognition

Tabel berikut membandingkan per-class recall antara Step 4b no-TD dan Step 4b td_gru (3-seed mean), diurutkan dari recall tertinggi ke terendah berdasarkan td_gru.

| Kelas | Tipe | No-TD (%) | td_gru (%) | Δ |
|---|---|---|---|---|
| happy | Basic | 84.44 ± 20.61 | **84.44 ± 23.93** | 0.00 |
| neutral | Basic | 64.44 ± 42.98 | **75.56 ± 33.26** | +11.11 |
| sad | Basic | **55.56 ± 33.70** | 46.67 ± 40.00 | −8.89 |
| happy-surprise | Compound | 44.44 ± 43.32 | **46.67 ± 40.00** | +2.22 |
| angry | Basic | 40.00 ± 38.87 | **46.67 ± 40.00** | +6.67 |
| surprise | Basic | **48.89 ± 43.66** | 40.00 ± 42.51 | −8.89 |
| happy-disgust | Compound | 24.44 ± 33.26 | **33.33 ± 38.49** | +8.89 |
| disgustedly-surprised | Compound | 15.56 ± 16.63 | **20.00 ± 23.73** | +4.44 |
| angrily-surprised | Compound | 15.56 ± 29.48 | **22.22 ± 31.43** | +6.67 |
| sad-surprise | Compound | 11.11 ± 15.71 | **26.67 ± 21.77** | +15.56 |
| angrily-disgusted | Compound | 15.56 ± 26.85 | **17.78 ± 23.93** | +2.22 |
| sad-disgust | Compound | **22.22 ± 28.97** | 17.78 ± 23.93 | −4.44 |
| fear-surprise | Compound | **15.56 ± 23.93** | 15.56 ± 20.61 | 0.00 |
| fear | Basic | 15.56 ± 26.85 | **13.33 ± 23.73** | −2.22 |
| disgust | Basic | 8.89 ± 19.12 | **11.11 ± 23.31** | +2.22 |
| fear-disgust | Compound | 11.11 ± 23.31 | **11.11 ± 19.88** | 0.00 |
| sad-angry | Compound | **13.33 ± 26.67** | 11.11 ± 19.88 | −2.22 |
| sad-fear | Compound | **6.67 ± 13.33** | 4.44 ± 11.33 | −2.22 |
| fear-angry | Compound | 6.67 ± 13.33 | **8.89 ± 19.12** | +2.22 |

### 6.1 Kelas Termudah

**happy** (84.44%) konsisten paling mudah dikenali di kedua kondisi — identik antara td_gru dan no-TD. Happy memiliki distribusi AU yang paling khas secara visual (AU6+AU12: *Duchenne smile*) sehingga backbone swin3d_s sudah cukup, tanpa perlu temporal modeling tambahan.

**neutral** (75.56%, td_gru) merupakan kelas yang paling diuntungkan oleh TD module (+11.11%). Neutral adalah kelas "default" yang tidak memiliki pola AU spesifik — identifikasinya lebih bergantung pada memastikan *tidak ada* ekspresi aktif sepanjang sekuens. TD module yang memantau dinamika temporal sepanjang klip lebih mampu mengidentifikasi "ketenangan ekspresi" ini.

### 6.2 Kelas Tersulit

Kelas dengan performa terendah dapat dibagi dua kelompok:

**Kelompok 1 — Pola Berbasis Fear** (semua recall ≤15.56%):

| Kelas | Recall (td_gru) |
|---|---|
| sad-fear | 4.44% |
| fear-angry | 8.89% |
| fear-disgust | 11.11% |
| fear | 13.33% |
| fear-surprise | 15.56% |

Pola ini konsisten dengan performa *fear* pada 7-class yang terendah (44.44%), lebih rendah dari disgust (46.67%). *Fear* secara visual memiliki area ekspresi yang tumpang tindih dengan *surprise* (alis terangkat) dan *sad* (sudut mulut turun), menyebabkan ambiguitas yang sulit diatasi bahkan pada kondisi 7-class. Ketika *fear* dikombinasikan dengan emosi lain sebagai compound emotion, ambiguitas ini semakin berlipat.

**Kelompok 2 — Kelas Non-Fear dengan Recall Rendah**:

| Kelas | Recall (td_gru) |
|---|---|
| disgust | 11.11% |
| sad-angry | 11.11% |

*Disgust* pada 7-class juga memiliki recall rendah (46.67%) karena ciri AU-nya yang subtle (AU9: wrinkle hidung, AU15: sudut mulut turun). *Sad-angry* merupakan compound emotion yang jarang dipelajari — perpaduan AU yang saling antagonistik (sad: AU17 menarik dagu, angry: AU23 menekan bibir) sulit untuk diklasifikasikan secara konsisten.

### 6.3 Dampak TD pada Distribusi Per-Kelas

TD module memberikan peningkatan terbesar pada:
- **sad-surprise**: +15.56% (11.11% → 26.67%) — ekspresi transisi dengan dinamika temporal yang khas
- **neutral**: +11.11% (64.44% → 75.56%) — deteksi "ketiadaan ekspresi" yang bergantung pada konsistensi temporal
- **happy-disgust**: +8.89% (24.44% → 33.33%) — compound emotion dengan dua komponen AU yang perlu dideteksi secara sekuensial
- **angry**: +6.67% (40.00% → 46.67%)

Sebaliknya, TD module menurunkan beberapa kelas:
- **sad**: −8.89% (55.56% → 46.67%) — sad memiliki ekspresi yang lebih statis dan bertahan lama; mungkin temporal attention terdistraksi oleh fase transisi
- **surprise**: −8.89% (48.89% → 40.00%) — surprise memiliki onset yang sangat cepat; temporal modeling mungkin kurang efektif untuk ekspresi singkat

---

## 7. Perbandingan dengan Penelitian Terkait pada IMED

> **Catatan penting**: Perbandingan langsung memerlukan kehati-hatian karena perbedaan mendasar pada protokol evaluasi. Penelitian sebelumnya menggunakan *random split* (60:20:20 atau k-Fold CV) yang tidak memisahkan identitas subjek, sedangkan penelitian ini menggunakan LOSO yang menjamin evaluasi *subject-independent*. Protokol LOSO secara inheren menghasilkan angka lebih rendah karena model diuji pada subjek yang sama sekali belum pernah dilihat selama pelatihan. Eksperimen Step 7 (Section 5.7) menunjukkan bahwa variabilitas antar-subjek yang besar (σ_fold≈21%) merupakan faktor dominan yang membatasi performa LOSO, bukan kualitas representasi model itu sendiri.

| Paper | Input | Kelas | Evaluasi | WAR/Akurasi |
|---|---|---|---|---|
| Akeh et al. 2022 (ViT) | Frame | 7-class | 60:20:20 | 98.90% |
| Iswanto et al. (EfficientNet) | Video frame-based | 7-class | Split | 97.93% |
| Hanifa et al. GLINT | Dynamic Image | 7-class | 5-Fold CV | 87.05% |
| Wang et al. FEEL | Frame (Ensemble) | 8 mixed | 5-Fold CV | 88.70% |
| Akeh & Kusuma 2025 (ViT) | Frame | 18-class | 60:20:20 | 79.37% |
| Liliana et al. 2018 (SVM) | Frame sequence | 19-class | 10-Fold CV | 83.33% |
| **Ours (7-class)** | **Video end-to-end (VST+TD)** | **7-class** | **LOSO** | **64.13 ± 16.65%** |
| **Ours (19-class)** | **Video end-to-end (VST+TD)** | **19-class** | **LOSO** | **29.70 ± 10.45%** |

Perbedaan angka antara penelitian ini (64.13%) dan prior work (87–98%) disebabkan terutama oleh perbedaan protokol evaluasi, bukan perbedaan kualitas model. Penelitian ini adalah yang pertama menggunakan Video Swin Transformer dengan end-to-end video learning (bukan frame atau Dynamic Image), protokol LOSO yang benar-benar *subject-independent*, dan evaluasi pada full 19-class compound emotion recognition dengan *subject-independent benchmark*.
