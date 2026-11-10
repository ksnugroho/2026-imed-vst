# Systematic Literature Review (SLR): Pengenalan Ekspresi Wajah dan Emosi Campuran Berbasis Dataset IMED

Dokumen ini menyajikan *Systematic Literature Review* (SLR) terhadap 10 penelitian ilmiah yang berfokus pada Pengenalan Ekspresi Wajah (*Facial Expression Recognition* / FER), khususnya yang memanfaatkan **Indonesian Mixed Emotion Dataset (IMED)** maupun kombinasinya dengan dataset internasional [4, 38, 59, 84, 105, 125, 150, 167, 208, 259].

---

## 1. Matriks Perbandingan SLR (10 Paper)

| No | Paper & Penulis | Dataset & Kelas Emosi | Input Model | Novelty / Kontribusi Utama | Metode / Arsitektur Model | Cara Evaluasi | Performance Metric (Hasil) | Limitasi / Kelemahan |
|---|---|---|---|---|---|---|---|---|
| **1** | **Akeh & Kusuma (2025)** [2, 7] | IMED (18 kelas: 12 mixed, 6 basic) [9, 10] | Gambar / Frame (256x256) [8, 11] | Evaluasi Transformer & ConvNeXt pada emosi campuran tanpa augmentasi berat [7, 8, 28] | ViT B-16, Swin v2 B, ConvNeXt Base [12] | Split 60:20:20, Stratified Sampling, Variasi Optimizer [10, 13] | **ViT B-16**: Akurasi 79.37%, F1 79.37% [23, 26]; ConvNeXt: 74.77% [25, 26]; Swin v2: 65.36% [24, 26] | Tanpa augmentasi gambar, Swin v2 underfit pada sampel kecil [22, 28] |
| **2** | **Liliana et al. (2018)** [38] | IMED (19 kelas: 6 basic, 12 mixed, 1 neutral) [36, 38] | Gambar & Video (Sekuens) [38, 41, 48] | Menerbitkan dataset emosi campuran Indonesia (IMED) pertama sebagai ground-truth benchmark [36, 38] | SVM, CRF, Hybrid SVM-CRF, AAM [47, 48, 51] | Validasi 12 Psikolog (Sign Test), 10-Fold CV untuk basic, Temporal sequence untuk mixed [46, 47, 48] | **SVM (frame)**: 84.52% [48]; **SVM-CRF**: 83.33% [48]; Pure CRF: 77.5% [48] | Data bersifat *posed* di studio, jumlah subjek terbatas (15 orang) [38, 44, 45] |
| **3** | **Hanifa et al. (GLINT)** [53] | IMED (7 kelas emosi dasar) [59] | Dynamic Image (2D terkompresi) [55, 59, 60] | Modul GLINT: gating adaptif tingkat input ($\alpha$) untuk memperkuat mikro-gerakan [56] | EfficientNet-B0 + GLINT [56, 60, 61] | Stratified 5-Fold CV, Paired t-test, Ablation study [59, 62, 65, 67] | **EfficientNet-B0 + GLINT**: Akurasi 87.05 ± 0.51%, Macro-F1 85.76% ($p=0.025$) [65] | Dataset laboratorium terkontrol kecil, tantangan pada emosi berkecepatan rendah (neutral) [70, 72] |
| **4** | **Nafis et al. (2020)** [149] | CK+, IMED, Video Real-World (7 kelas dasar) [83, 84, 92] | Gambar & Video Real-time [85, 87, 92] | Deteksi wajah & klasifikasi ekspresi *single-stage* pose bervariasi dengan YOLOv3 [80, 81, 97] | YOLOv3 (Darknet-53) vs Faster R-CNN [81, 96] | Variasi Channel & Learning Rate, Testing per dataset, Majority Voting video [90, 91, 92] | **YOLOv3 (Overall)**: 87% [90]; CK+: 94% [91]; IMED: 79% [91]; **Video Real-Time**: 73% [92] | Performa menurun pada video tidak terkontrol saat subjek kurang ekspresif [93, 97] |
| **5** | **Jala et al. (2020)** [101] | IMED (5 kelas emosi dasar) [105, 106] | Gambar (48x48) & Real-time Webcam [106, 108, 112] | Arsitektur CNN kustom 4-layer + preprocessing Viola-Jones real-time [101, 107, 110, 115] | Custom 4-Layer CNN [110] | Eksperimen Epoch (100-300) & LR (0.001-0.1), Pengujian Real-Time Webcam [113, 115, 118] | **Validasi IMED**: 99.76% (LR 0.001, Epoch 300) [118, 119]; **Real-Time Webcam**: 93.63% [101, 115, 119] | Mensyaratkan kondisi real-time sangat terkontrol (pencahayaan, sudut wajah) [115] |
| **6** | **Iswanto et al. (Dynamic Image)** [122] | IMED (7 kelas emosi dasar) [132, 133] | Dynamic Image, Static Image, Video [122, 126, 133] | Menerapkan Dynamic Image (Rank Pooling) pertama kali pada dataset FER Indonesia [122, 126, 127] | EfficientNet-B0 [133, 134] | Komparasi langsung 3 tipe input (Dynamic, Static, Video) pada arsitektur sama [135, 136] | **Dynamic Image**: 94.28% [136]; Static Image: 93.37% [136]; **Video**: 97.93% [136] | Terjadi *data imbalance* antar kelas emosi, kualitas terikat segmentasi [122, 132, 133] |
| **7** | **Akeh et al. (2022 - ViT)** [146, 147] | IMED (7 kelas emosi dasar) [147, 151] | Gambar (Frame IMED) [151] | Fine-tuning ViT + 1 Linear Layer untuk emosi dasar & integrasi Flutter [147, 150, 158, 159] | Vision Transformer (ViT) pre-trained + 1 Linear Layer [150] | Split 60:20:20, Grid Tuning Hyperparameter (Epoch, Batch, LR, Optimizer) [152, 153, 154] | **Test Accuracy**: 98.9%, Validation Loss 0.044 (hanya 4 epoch) [157, 158, 159]; Prediksi 21.5% lebih cepat [147] | Emosi Fear & Disgust paling sulit diprediksi, tidak menguji kelas emosi campuran [147, 150, 159] |
| **8** | **Mahastama et al. (2024)** [166] | FER2013, CK+, IMED (4 kelas dasar) [167, 176, 180] | Gambar Statis Grayscale (48x48) [178] | Stacked CNN (1024 filter) + callback otomatis untuk dataset gabungan multiras [170, 175, 182, 184, 190] | Stacked CNN (5 Blok Konvolusi) [181, 182, 183, 190] | 9 Skenario Pengujian dataset tunggal & gabungan, Confusion Matrix, ROC-AUC [187, 191, 192, 193] | **Validasi Gabungan (Skenario 9)**: 79.59% [190, 191, 195]; **Test Rata-Rata**: 71.5% [168, 196] | Performa menurun pada gambar bernoise, kartun, atau wajah tertutup [194, 195] |
| **9** | **Wang et al. (FEEL - PeerJ CS)** [201] | FER2013, IMED (7 dasar, 8 mixed) [201, 208, 227] | Gambar (224x224) [210, 213] | Dual-Stage Ensemble (ESER & ECER) + SSM untuk memisahkan emosi dasar & campuran [201, 203, 211, 212] | Ensemble (DenseNet, VGG, ResNet, Xception, ViT) + XGBoost + SE Block [201, 206, 212, 217] | Stratified 5-Fold CV, Split 70:15:15, Paired t-test, Pruning Assessment [222, 223, 225] | **Emosi Dasar (ESER)**: 95.0% [201, 223]; **Emosi Campuran (ECER)**: 88.7% [201, 224] | Kompleksitas komputasi tinggi akibat penggunaan banyak model deep learning [225, 226, 229] |
| **10** | **Liliana & Basaruddin (2019)** [243] | CK+, IMED, JAFFE, DISFA (6 kelas dasar) [243, 259, 260] | Gambar Statis Frontal [243, 251] | Framework Fuzzy Emotion Explainable AI berbasis aturan pakar psikolog [243, 247, 251, 256, 257] | Active Appearance Model (AAM) + Geometric Feature + Fuzzy Sugeno (FEC) [251, 254, 257] | Pengujian FFC komponen linguistik pada CK+, Pengujian FEC pada 4 dataset [259, 260] | **Akurasi FFC (CK+)**: 97.9% [260, 261]; **Akurasi Fuzzy IMED**: 87.0% [243, 263]; CK+: 90% [243, 263] | Nilai recall lebih rendah dari precision karena sistem berbasis aturan tidak belajar dari data [262] |

---

## 2. Analisis Detail Per Paper

### Paper 1: Mixed Emotion Recognition through Facial Expression using Transformer-based and ConvNeXt Models
* **Penulis & Tahun**: Limas Jaya Akeh & G. P. Kusuma (2025) [2, 7]
* **Dataset & Kelas Emosi**: Dataset **IMED** [4, 9, 10]. Menggunakan **18 kelas emosi** yang terdiri dari **12 emosi campuran** (*mixed emotions*) dan **6 emosi dasar** (*Angry, Disgust, Fear, Happy, Sad, Surprised*) [9, 10]. Kelas *Neutral* dikeluarkan karena jumlah datanya sangat sedikit [10]. Total data terdiri dari 281 video dan 30,254 gambar [9].
* **Input Model**: **Gambar / Frame** (256x256 pixel) yang di-crop dari segmen wajah hasil deteksi MTCNN [8, 11].
* **Novelty**: Memodifikasi *linear head classifier* dari arsitektur Transformer mutakhir (Vision Transformer dan Swin Transformer v2) serta ConvNeXt untuk menguji kemampuannya dalam mengklasifikasikan emosi campuran kompleks pada populasi Indonesia tanpa dependensi augmentasi data berat [7, 8, 12, 28].
* **Metode / Model**: Vision Transformer (ViT B-16), Swin Transformer v2 (Swin v2 B), dan ConvNeXt Base [12]. Optimizer diuji dengan Adam, SGD, dan SGD dengan *decay learning rate* (StepLR) [13].
* **Cara Evaluasi**: Pembagian data *Train:Validation:Test* sebesar 60:20:20 menggunakan *stratified sampling* [10]. Evaluasi performa menggunakan Weighted Average Precision, Weighted Average Recall, Weighted Average F1-Score, dan Overall Accuracy [23].
* **Performance Metric (Hasil)**:
  * **Vision Transformer (ViT B-16)**: Akurasi Pengujian **79.37%**, Precision **79.49%**, Recall **79.37%**, F1-Score **79.37%** [23, 26].
  * **ConvNeXt Base**: Akurasi Pengujian **74.77%**, Precision **75.31%**, Recall **74.77%**, F1-Score **74.71%** [25, 26].
  * **Swin Transformer v2**: Akurasi Pengujian **65.36%**, Precision **68.12%**, Recall **65.36%**, F1-Score **65.59%** [24, 26].
* **Limitasi / Kelemahan**: Terbatas pada teknik *preprocessing* dasar (crop & resize) tanpa augmentasi gambar [28]; keterbatasan MTCNN menyebabkan beberapa wajah tidak terdeteksi [11]; Swin v2 mengalami *underfitting* akibat resolusi dan sampel terbatas pada label tertentu [22]; tuning hyperparameter terbatas oleh daya komputasi [29].
* **Kesimpulan**: Vision Transformer (ViT B-16) terbukti paling unggul di antara ketiga model dalam mengenali 18 kelas emosi campuran pada dataset IMED [27].

---

### Paper 2: The Indonesian Mixed Emotion Dataset (IMED): A Facial Expression Dataset for Mixed Emotion Recognition
* **Penulis & Tahun**: Dewi Yanti Liliana, Tjan Basaruddin, & Imelda Ika Dian Oriza (2018) [38]
* **Dataset & Kelas Emosi**: **IMED** (*Indonesian Mixed Emotion Dataset*) [36, 38]. Mencakup **19 kelas emosi** (6 emosi dasar, **12 emosi campuran**, dan 1 netral) yang diperagakan oleh 15 subjek warga negara Indonesia dari berbagai etnis [36, 38, 44]. Total data berjumlah 570 video dan 66,819 frame gambar (720x480 pixel) [38].
* **Input Model**: **Gambar Statis** (frame-by-frame) dan **Video / Sekuens Gambar** [38, 41, 48].
* **Novelty**: Mengembangkan dan mempublikasikan dataset wajah khusus emosi campuran terorientasi etnis Indonesia (IMED) sebagai *ground-truth benchmark* pertama untuk pengenalan emosi non-dasar [36, 38].
* **Metode / Model**: Active Appearance Model (AAM) untuk ekstraksi titik wajah, Support Vector Machine (SVM), Conditional Random Field (CRF), dan model hibrida SVM-CRF [47, 48, 51].
* **Cara Evaluasi**: Validasi kelayakan dataset oleh 12 pakar psikologi menggunakan *Sign Test* [46]. Evaluasi komputasional menggunakan 10-Fold Cross Validation untuk emosi dasar, serta pengujian klasifikasi temporal untuk sekuens emosi campuran [47, 48].
* **Performance Metric (Hasil)**:
  * **SVM (image sequence, n=7)**: Akurasi **84.52%** [48].
  * **SVM-CRF (hibrida frame & sekuens)**: Akurasi **83.33%** [48].
  * **Pure CRF**: Akurasi **77.5%** [48].
* **Limitasi / Kelemahan**: Pengambilan ekspresi wajah bersifat *posed* (diperagakan di studio terkontrol) bukan *spontaneous* [38, 45]; jumlah subjek pemeraga relatif kecil (15 orang) [36, 44].
* **Kesimpulan**: Dataset IMED berhasil dikembangkan dan divalidasi, serta terbukti dapat secara andal dibedakan oleh pengklasifikasi mesin dengan akurasi *benchmark* mencapai 83–84% [36, 48, 49].

---

### Paper 3: Enhancing Motion Sensitivity in Dynamic Images for Facial Expression Recognition through GLINT: A Learnable Input-Level Gating Mechanism
* **Penulis & Tahun**: Aida Hanifa, Ilham Syahrul Ramadhan, & Irene Anindaputri Iswanto [53]
* **Dataset & Kelas Emosi**: Dataset **IMED** [59]. Menggunakan **7 kelas emosi dasar** (*anger, disgust, fear, happiness, sadness, surprise, neutral*) dari 105 sampel video yang diekstraksi menjadi 1,961 representasi *Dynamic Image* (disaring menjadi 1,855 sampel valid) [59, 60].
* **Input Model**: **Dynamic Image** (representasi gambar statis RGB 2D 224x224 yang mengompresi pergerakan temporal sekuens video) [55, 59, 60].
* **Novelty**: Mengusulkan **GLINT** (*Gated Learnable Input Mechanism*), yaitu mekanisme *gating* adaptif tingkat input menggunakan parameter terpelajar ($\alpha$) untuk memperkuat area mikro-gerakan (*micro-motion*) sebagai *motion prior* sebelum ekstraksi fitur CNN [56].
* **Metode / Model**: Arsitektur backbone EfficientNet-B0 yang diintegrasikan dengan modul input-gating GLINT [56, 60, 61].
* **Cara Evaluasi**: *Stratified 5-Fold Cross-Validation* [59]. Uji signifikansi statistik (*paired t-test*) dan studi ablasi (*Mask-Only, Alpha-Only, GLINT Full*) [59, 62, 65, 67].
* **Performance Metric (Hasil)**:
  * **EfficientNet-B0 + Pure DI (Baseline)**: Akurasi **84.80 ± 1.43%**, Macro-F1 **83.06 ± 1.12%** [65].
  * **EfficientNet-B0 + GLINT (Ours)**: Akurasi **87.05 ± 0.51%**, Macro-F1 **85.76 ± 0.80%** (peningkatan akurasi 2.25%, signifikan secara statistik $p = 0.025$) [65]. Overhead waktu inferensi sangat kecil (+2.8% / +0.23 ms) [66, 67].
* **Limitasi / Kelemahan**: Terbatas pada dataset IMED yang berukuran kecil dan berbasis kondisi laboratorium [72]; performa masih menantang pada emosi dengan gerakan rendah seperti emosi *neutral* [70, 72].
* **Kesimpulan**: Modul GLINT terbukti meningkatkan sensitivitas terhadap pola mikro-gerakan pada *Dynamic Image* sehingga meningkatkan akurasi FER tanpa menambah beban komputasi signifikan [70, 71].

---

### Paper 4: Facial Expression Recognition on Video Data with Various Face Poses Using Deep Learning
* **Penulis & Tahun**: Ayas Faikar Nafis, Dini Adni Navastara, & Anny Yuniarti (2020) [149]
* **Dataset & Kelas Emosi**: Gabungan dataset **CK+** (210 gambar), **IMED** (175 gambar), dan **Data Video Real-World** (8 subjek mahasiswa ITS dengan variasi pose wajah) [83, 84, 91, 92]. Fokus pada **7 kelas emosi dasar** [78, 85].
* **Input Model**: **Gambar** (frame 416x416) dan **Video Real-time** dengan variasi pose wajah [85, 87, 92].
* **Novelty**: Mengimplementasikan algoritma deteksi objek *single-stage* YOLOv3 untuk melakukan deteksi wajah sekaligus klasifikasi ekspresi wajah secara langsung dalam satu tahap pada data video berpose bervariasi di lingkungan tidak terkontrol [80, 81, 97].
* **Metode / Model**: YOLOv3 berbasis Darknet-53 [81]. Preprocessing ekstraksi *bounding box* wajah menggunakan MTCNN [87]. Komparasi dilakukan terhadap Faster R-CNN [96].
* **Cara Evaluasi**: Pengujian variasi kanal gambar (1 vs 3 channel), variasi *learning rate* (0.1, 0.01, 0.001), pengujian per dataset, serta pengujian data video real-time dengan metode *majority voting* per video [90, 91, 92].
* **Performance Metric (Hasil)**:
  * **Akurasi YOLOv3 (Overall)**: **87%** (Channel 3, Learning Rate 0.01) [90].
  * **Akurasi per Dataset**: CK+ = **94%**, IMED = **79%** [91, 149].
  * **Pengujian Video Real-time**: **73%** video terklasifikasi benar (41 dari 56 video) menggunakan Model 2 [92].
  * **Komparasi vs Faster R-CNN**: YOLOv3 mencapai akurasi **87%** dengan waktu uji **44.85 detik**, mengungguli Faster R-CNN (akurasi **72%**, waktu uji **456.34 detik**) [94, 96].
* **Limitasi / Kelemahan**: Performa menurun pada data video lingkungan tidak terkontrol saat subjek kurang ekspresif atau ekspresi antar kelas mirip [93, 97].
* **Kesimpulan**: Pendekatan deteksi satu tahap YOLOv3 sangat efisien dan akurat untuk mengenali ekspresi wajah dari sekuens gambar/video dengan variasi pose [96, 97].

---

### Paper 5: Implementation of Convolutional Neural Network (CNN) Algorithm for Classification of Human Facial Expression in Indonesia
* **Penulis & Tahun**: Aqil Bayu Jala, Tito Waluyo Purboyo, & Ratna Astuti Nugrahaeni (2020) [101]
* **Dataset & Kelas Emosi**: Dataset **IMED** [101, 105]. Menggunakan **5 kelas emosi dasar**: *Angry, Happy, Neutral, Sad, Surprise* [105, 106]. Jumlah sampel terdiri dari 1,680 data pelatihan dan 420 data validasi [106].
* **Input Model**: **Gambar** (48x48 pixel grayscale) dan pengujian **Webcam Real-time** [106, 108, 112].
* **Novelty**: Penerapan arsitektur CNN kustom 4-layer konvolusi dengan tahapan *preprocessing* khusus (Viola-Jones, cropping, grayscaling, resizing) dan augmentasi data untuk deteksi ekspresi wajah orang Indonesia secara *real-time* [101, 107, 110, 115].
* **Metode / Model**: Convolutional Neural Network (CNN) kustom 4-layer [110]. Preprocessing wajah menggunakan Haar Cascade Viola-Jones (`haarcascade_frontalface_default.xml`) [107].
* **Cara Evaluasi**: Eksperimen variasi jumlah Epoch (100, 200, 300) dan Learning Rate (0.001, 0.01, 0.1) pada data validasi IMED, diikuti pengujian aplikasi *real-time* via webcam pada 110 ekspresi uji [113, 115, 118].
* **Performance Metric (Hasil)**:
  * **Akurasi Validasi Model (IMED)**: **99.76%** (Learning Rate 0.001, Epoch 300) [118, 119].
  * **Akurasi Pengujian Real-Time Webcam**: **93.63%** dari 110 pengujian ekspresi [101, 115, 119].
* **Limitasi / Kelemahan**: Pengujian real-time mensyaratkan kondisi lingkungan terkontrol (wajah lurus 180°, jarak 30cm–1m, pencahayaan 1,500–2,250 Lumens) [115]; akurasi mengalami penurunan saat diuji pada wajah di luar dataset IMED [6].
* **Kesimpulan**: Model CNN kustom mampu mengingat pola ekspresi IMED hingga akurasi 99.76% dan melakukan klasifikasi real-time dengan akurasi 93.63% pada kondisi terkontrol [101, 119].

---

### Paper 6: Implementation of Dynamic Image for Facial Expression Recognition on Indonesian Facial Expression Dataset
* **Penulis & Tahun**: Irene Anindaputri Iswanto, Andry Chowanda, Haryono Soeparno, & Widodo Budiharto [122]
* **Dataset & Kelas Emosi**: Dataset **IMED** [122, 125, 131]. Menggunakan **7 kelas emosi dasar**: *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral* [132, 133]. Terdiri dari 6,034 *Dynamic Images* hasil segmentasi 570 video IMED (window duration 10, stride 2) [132].
* **Input Model**: **Dynamic Image**, **IMED Static Image**, dan **IMED Video** (sekuens gambar) [122, 126, 133].
* **Novelty**: Implementasi metode *Dynamic Image* (mengompresi dinamika waktu video menjadi satu gambar statis berbasis Rank Pooling) untuk pertama kalinya pada FER dataset Indonesia (IMED) guna menghemat kompleksitas dimensi data [122, 126, 127].
* **Metode / Model**: Arsitektur EfficientNet-B0 (memanfaatkan blok MBConv dengan *compound scaling*) [133, 134]. Preprocessing cropping wajah via MTCNN [131, 132].
* **Cara Evaluasi**: Membandingkan secara langsung performa EfficientNet-B0 saat dilatih menggunakan tiga tipe input berbeda (*Dynamic Image*, *Static Image*, dan *Video*) pada dataset IMED [135, 136].
* **Performance Metric (Hasil)**:
  * **Dynamic Image**: Akurasi **94.28%**, Precision 94.26%, Recall 93.83%, F1-Score 94.02% [136].
  * **Static Image**: Akurasi **93.37%**, Precision 93.53%, Recall 93.76%, F1-Score 93.61% [136].
  * **Video**: Akurasi **97.93%**, Precision 98.12%, Recall 97.84%, F1-Score 97.96% [136].
* **Limitasi / Kelemahan**: Terdapat ketidakseimbangan jumlah data (*data imbalance*) antar kelas emosi [122, 132, 133]; kualitas *Dynamic Image* bergantung pada proses segmentasi video [122].
* **Kesimpulan**: *Dynamic Image* mengungguli *Static Image* (94.28% vs 93.37%) dan mendekati akurasi input *Video* (97.93%) dengan kebutuhan data pelatihan dan kompleksitas dimensi yang jauh lebih efisien [122, 136, 137].

---

### Paper 7: Modelling Emotions Recognition from Facial Expression using Vision Transformer with IMED Dataset
* **Penulis & Tahun**: Limas Jaya Akeh, Ronaldo Kenny Chandra, William Loo, & Rhio Sutoyo (2022) [146, 147]
* **Dataset & Kelas Emosi**: Dataset **IMED** [147, 150]. Berfokus pada **7 kelas emosi dasar (single emotions)**: *Angry, Sad, Happy, Neutral, Surprised, Disgust, Fear* [147, 151]. Total data berjumlah 8,947 gambar [151, 152].
* **Input Model**: **Gambar** (frame wajah ter-crop & resize dari sekuens video IMED) [151].
* **Novelty**: Mengadaptasi dan melakukan *fine-tuning* pada arsitektur Vision Transformer (ViT) *pre-trained* dengan penambahan 1 *Linear Layer Classifier* untuk emosi dasar IMED, serta mengintegrasikannya ke dalam aplikasi mobile/web berbasis Flutter [147, 150, 158, 159].
* **Metode / Model**: Vision Transformer (ViT) *pre-trained* + 1 Linear Layer [150]. Preprocessing via *ViTFeatureExtractor* (cropping, resizing, normalisasi) [151]. Grid search hyperparameter tuning [152, 153].
* **Cara Evaluasi**: Pembagian data *Train:Validation:Test* sebesar 60:20:20 [152]. Eksperimen grid tuning (augmentasi, dropout, learning rate, batch size, optimizer) menggunakan PyTorch CrossEntropyLoss [153, 154]. Komparasi dengan CNN, YOLOv3, dan ARCH-1 [158].
* **Performance Metric (Hasil)**:
  * **Vision Transformer (ViT)**: Test Accuracy **98.9%**, Validation Loss **0.044** (Learning Rate 0.00001, Batch Size 16, dan 4 Epochs) [157, 158, 159].
  * **Kecepatan Prediksi**: 21.5% lebih cepat dibanding rata-rata model sejenis [147].
  * **Komparasi**: Mengungguli ARCH-1 (96.4%) dan YOLOv3 (79%), serta berakurasi setara CNN (99.76%) dalam epoch yang jauh lebih sedikit (4 epoch vs 300 epoch) [158, 159].
* **Limitasi / Kelemahan**: Emosi *Fear* dan *Disgust* paling sulit diprediksi akibat kemiripan visual pada alis, mata, dan mulut [147, 159]; tidak menguji 12 kelas emosi campuran yang ada pada IMED [150, 151].
* **Kesimpulan**: Vision Transformer (ViT) sangat efisien dan akurat untuk klasifikasi 7 emosi dasar pada IMED dengan akurasi 98.9% hanya dalam 4 epoch pelatihan [147, 158, 159].

---

### Paper 8: Facial Expression Classification System Using Stacked CNN
* **Penulis & Tahun**: Aditya Wikan Mahastama, Edwin Mahendra, Antonius Rachmat Chrismanto, Maria Nila Anggia Rini, & Andhika Galuh Prabawati (2024) [166]
* **Dataset & Kelas Emosi**: Kombinasi dataset **FER2013, CK+, dan IMED** [167, 176]. Diseleksi menjadi **4 kelas emosi dasar**: *Happy, Angry, Neutral, Sad* [180]. Pembagian data *Train:Validation:Test* sebesar 80:18:2 [180].
* **Input Model**: **Gambar Statis** Grayscale ukuran 48x48 pixel [178].
* **Novelty**: Merancang arsitektur *Stacked CNN* yang diperdalam dengan penambahan blok konvolusi ekstra (filter 1024) serta menerapkan mekanisme *callback* otomatis (*Early Stopping*, *ReduceLROnPlateau*, *ModelCheckpoint*) untuk meningkatkan generalisasi pada gabungan dataset multiras (Barat dan Indonesia) [170, 175, 182, 184, 190].
* **Metode / Model**: Stacked CNN (5 blok konvolusi dengan filter 256, 512, 384, 192, 512 + penambahan filter 1024, Batch Normalization, Max Pooling, Dropout 0.5, Dense 256, Softmax) [181, 182, 183, 190]. Preprocessing IMED menggunakan Haar Cascade Classifier [177, 178].
* **Cara Evaluasi**: 9 skenario eksperimen kombinasi dataset (tunggal vs gabungan) dan variasi blok konvolusi [187, 191]. Evaluasi menggunakan Accuracy, Precision, Recall, F1-Score, Confusion Matrix, dan ROC-AUC [185, 191, 192, 193].
* **Performance Metric (Hasil)**:
  * **Akurasi Validasi Skenario 9 (Gabungan FER2013 + CK+ + IMED dengan filter 1024)**: **79.59%**, Validation Loss 0.6206 [190, 191, 195].
  * **Rata-Rata Test Set (Data Uji Gabungan)**: **71.5%** (Precision 71.5%, Recall 71.5%, F1-Score 71.5%) [168, 196].
  * **Akurasi Validasi Dataset Individual**: Mencapai **97–98%** jika diuji pada CK+ atau IMED secara terpisah [191, 196].
  * **ROC-AUC**: Happy AUC 0.97, Angry AUC 0.90, Sad AUC 0.90, Neutral AUC 0.87 [194].
* **Limitasi / Kelemahan**: Performa menurun pada gambar bernoise (pencahayaan buruk, wajah tertutup tangan/watermark, atau gambar kartun/non-wajah) [194, 195]; variabilitas tinggi pada FER2013 menurunkan performa saat digabung dengan IMED/CK+ [191].
* **Kesimpulan**: Menggabungkan dataset lokal (IMED) dan internasional (CK+, FER2013) ke dalam Stacked CNN yang lebih dalam meningkatkan kapasitas ekstraksi fitur dan generalisasi lintas populasi [195, 196].

---

### Paper 9: FEEL: Fast and Effective Emotion Labeling, A dual ensemble approach for FER
* **Penulis & Tahun**: Shuiping Wang (PeerJ CS) [201]
* **Dataset & Kelas Emosi**: Gabungan dataset **FER2013 dan IMED** [201, 208]. Tahap 1 (ESER) mengklasifikasikan **7 emosi dasar**, sedangkan Tahap 2 (ECER) mengklasifikasikan **8 emosi campuran / blended emotions** (*Bittersweet Smile, Nostalgic Look, Melancholic Grin, Grateful Sob, Regretful Smile, Hopeful Despair, Contented Sadness, Joyful Tremble*) [201, 208, 227]. Total 3,750 gambar split 70:15:15 [213, 221, 222].
* **Input Model**: **Gambar** (224x224 grayscale/RGB via *bicubic interpolation*) [210, 213].
* **Novelty**: Kerangka kerja **Dual-Stage Ensemble (FEEL)** ber-modul Squeeze-and-Excitation (SE). ESER mengklasifikasikan emosi dasar; jika skor kepastian meta-learner < 50%, data diteruskan ke ECER yang diintegrasikan dengan *Secondary Statistical Module* (SSM) untuk mengenali emosi campuran [201, 203, 211, 212, 213].
* **Metode / Model**:
  * **ESER (Stage 1)**: Base models DenseNet169, VGG16, ResNet50 + XGBoost meta-learner [201, 206, 211].
  * **ECER (Stage 2)**: Base models Xception, ViT, DenseNet221 + XGBoost meta-learner + SSM (bobot 60% pengenalan langsung & 40% statistik) [201, 212, 213]. Modul SE terpasang pada base models [201, 217].
* **Cara Evaluasi**: *Stratified 5-Fold Cross-Validation*, split 70:15:15 [222]. Uji signifikansi statistik *paired t-test* ($p < 0.01$) [223]. Analisis efisiensi metode *pruning* [225, 226].
* **Performance Metric (Hasil)**:
  * **ESER (Emosi Dasar)**: Akurasi **95.0%**, Precision 0.95, Recall 0.95, F1-Score 0.95, AUC 0.95 [201, 211, 223].
  * **ECER (Emosi Campuran)**: Akurasi **88.7%**, Precision 0.88, Recall 0.88, F1-Score 0.88 [201, 224].
  * **Pruning Approach**: Mengeksklusi DenseNet dan Xception menghemat waktu komputasi 21% dengan penurunan akurasi hanya 7.05% (ESER 87%, ECER 83%) [226, 228].
* **Limitasi / Kelemahan**: Tingginya kompleksitas dan beban komputasi karena melibatkan banyak arsitektur deep learning secara bersamaan [225, 226, 229].
* **Kesimpulan**: Kerangka kerja Dual Ensemble berhasil memisahkan dan meningkatkan akurasi klasifikasi emosi dasar (95%) dan emosi campuran (88.7%) secara signifikan [201, 229, 230].

---

### Paper 10: The Fuzzy Emotion Recognition Framework Using Semantic-Linguistic Facial Features
* **Penulis & Tahun**: Dewi Yanti Liliana & T. Basaruddin (2019) [243]
* **Dataset & Kelas Emosi**: Dataset **CK+** (304 gambar), **IMED**, **JAFFE**, dan **DISFA** [243, 259, 260]. Mengevaluasi **6 kelas emosi dasar** (*angry, disgust, fear, happy, sad, surprise*) untuk menghasilkan nilai intensitas & derajat ambiguitas emosi (*fuzzy emotion degree*) [243, 245, 253, 260].
* **Input Model**: **Gambar Statis Wajah Tampak Depan** (*Static Frontal Face Image*) tanpa oklusi [243, 251].
* **Novelty**: Membangun kerangka kerja *Fuzzy Emotion Recognition* berprinsip *Explainable AI* yang menerjemahkan fitur geometris wajah ke dalam fitur linguistik (FFC) berdasarkan aturan pakar psikologi, lalu dievaluasi melalui *Fuzzy Inference System* (Sugeno) untuk menghasilkan derajat intensitas emosi [243, 247, 251, 256, 257].
* **Metode / Model**: Active Appearance Model (AAM) untuk deteksi titik wajah $\rightarrow$ 10 parameter geometris *Fuzzy Facial Component* (FFC) $\rightarrow$ 6 pengklasifikasi *Fuzzy Emotion Classification* (FEC) independen berbasis Fuzzy Sugeno dengan 42–83 aturan pakar per kelas emosi [251, 254, 256, 257].
* **Cara Evaluasi**: Dua skenario: 1) Pengujian akurasi penentuan kondisi linguistik komponen wajah (FFC) pada CK+ (304 gambar); 2) Pengujian akurasi pengenalan *fuzzy emotion* (FEC) pada CK+, JAFFE, DISFA, dan IMED [259, 260].
* **Performance Metric (Hasil)**:
  * **Akurasi Komponen Linguistik Wajah (FFC) pada CK+**: Akurasi **0.979 (97.9%)**, Precision 0.980, Recall 0.951, F1-Score 0.965 [260, 261].
  * **Akurasi Fuzzy Emotion Recognition pada IMED**: **0.87 (87.0%)** [243, 263].
  * **Akurasi pada Dataset Lain**: CK+ = **0.90 (90.0%)**, JAFFE ~85–90%, DISFA ~80–87% [243, 263].
* **Limitasi / Kelemahan**: Sifat sistem berbasis aturan (*rule-based*) menyebabkan nilai *recall* cenderung lebih rendah dari *precision* karena tidak belajar dari distribusi data seperti *generative machine learning* [262]; sangat bergantung pada keakuratan deteksi titik wajah AAM [262].
* **Kesimpulan**: Kerangka kerja fuzzy berbasis pengetahuan pakar psikologi efektif mengatasi ambiguitas ekspresi wajah dan menghasilkan derajat intensitas emosi (*fuzzy emotion degree*) yang transparan dengan akurasi 87% pada IMED [243, 257, 262, 264].

---

## 3. Analisis Sintesis Lintas Studi

1. **Representasi Input (Gambar Statis vs Video vs Dynamic Image)**:
   * Input **Video** memberikan akurasi tertinggi (mencapai **97.93%** pada EfficientNet-B0) [136] karena mempertahankan dinamika temporal penuh, namun membutuhkan ukuran data dan dimensi komputasi yang besar [126, 137].
   * **Dynamic Image** (mengompresi video menjadi 2D RGB statis via Rank Pooling) terbukti menjadi kompromi optimal: mengungguli Gambar Statis (**94.28% vs 93.37%**) dan mendekati performa Video dengan hanya menggunakan sekitar setengah jumlah data pelatihan [122, 136, 137].
   * Integrasi modul **GLINT** pada Dynamic Image semakin meningkatkan sensitivitas terhadap mikro-gerakan hingga akurasi **87.05%** [65].

2. **Perkembangan Arsitektur Model**:
   * Metode konvensional/pakar (*SVM, Fuzzy Sugeno*) memberikan interpretabilitas (*Explainable AI*) dengan akurasi 83–87% pada IMED [48, 243, 257].
   * Model **CNN Kustom / Stacked CNN** mampu mengingat pola dataset homogen hingga >99% validasi, namun akurasinya turun menjadi ~71–79% saat diuji pada gabungan dataset multiras (FER2013 + CK+ + IMED) [118, 168, 191].
   * Arsitektur **Vision Transformer (ViT)** terbukti sangat superior untuk emosi dasar (Akurasi **98.9%** hanya dalam 4 epoch) [158, 159], dan mencapai akurasi **79.37%** saat diuji pada 18 kelas emosi campuran kompleks tanpa augmentasi data [23, 26].
   * Pendekatan **Dual-Stage Ensemble (FEEL)** berhasil mengatasi ambiguitas emosi campuran dengan memisahkan tahap klasifikasi emosi dasar (ESER: **95%**) dan emosi campuran (ECER: **88.7%**) [201].

3. **Kompleksitas Klasifikasi: Emosi Dasar vs Emosi Campuran**:
   * Pengenalan **Emosi Dasar (5–7 kelas)** secara konsisten mencapai akurasi sangat tinggi (87% – 98.9%) di berbagai arsitektur [65, 90, 118, 136, 158, 201].
   * Pengenalan **Emosi Campuran / Mixed Emotions (18–19 kelas)** menghadirkan tantangan ambiguitas spasial-temporal yang jauh lebih tinggi [4, 38], di mana model Transformer tunggal mencapai ~79.37% [26] dan Dual-Stage Ensemble mencapai ~88.7% [201].
