**Catatan transkripsi — bukan bagian naskah tesis.** Berkas ini adalah salinan kata demi kata dari revisi naskah yang ditulis penulis (`TATATA.docx`, revisi dosen pembimbing, 2026-10-04), tanpa perbaikan apa pun. Kesalahan penomoran yang masih ada (§3.2.1 dua kali, Tabel 3.6 dua kali, rujukan Tabel 3.9) sengaja dipertahankan. Penanda `<Tabel 3.1>` dan sejenisnya adalah penanda posisi objek dari penulis; `![](...)` adalah gambar yang disertakan penulis dalam berkas Word.

Berkas ini adalah **bagian 1 dari 3**: Abstrak + BAB 1 + BAB 2 + BAB 3.

---

# BAB 1
# PENDAHULUAN

## 1.1. Latar Belakang

Perkembangan transformasi digital sektor keuangan telah mendorong peningkatan volume transaksi elektronik sekaligus memperluas ancaman kecurangan finansial (financial fraud). Laporan Association of Certified Fraud Examiners (ACFE) menunjukkan bahwa organisasi global kehilangan rata-rata 5% dari pendapatan tahunan mereka akibat fraud dengan kerugian median per kasus mencapai USD 145.000 (Association of Certified Fraud Examiners, 2024). Di sisi lain, kompleksitas metode financial fraud terus berkembang sehingga pendekatan deteksi tradisional berbasis rule-based menjadi semakin tidak memadai karena keterbatasannya dalam menangkap pola anomali yang dinamis dan terus beradaptasi (Hilal dkk, 2022). Maka dari itu, pendekatan berbasis machine learning (ML) telah digunakan secara luas untuk mendeteksi pola fraud melalui analisis data historis transaksi.

Adapun sebagian besar implementasi ML pada fraud detection masih bersifat terpusat (centralized) yang mengharuskan agregasi data mentah dari berbagai sumber institusi ke satu titik terpusat. Implementasi ini menimbulkan masalah terkait privasi, kerahasiaan data nasabah, dan kepatuhan terhadap regulasi seperti General Data Protection Regulation (GDPR) serta Undang-Undang Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi (PDP). Hal ini menghambat kolaborasi antar lembaga keuangan untuk menghasilkan sumber data yang global untuk membangun model deteksi fraud yang robust.

Federated Learning (FL) muncul sebagai pendekatan alternatif yang memungkinkan beberapa pihak melatih model secara kolaboratif tanpa memindahkan data mentah dari masing-masing sumber (McMahan dkk, 2017). Dalam pendekatan ini, pelatihan model dilakukan secara lokal pada setiap client dan hanya parameter atau model baru yang dipindahkan ke server pusat untuk diagregasi. Pendekatan ini relevan dalam sektor keuangan karena memungkinkan kolaborasi antar institusi dengan menghindari perpindahan data mentah selama pelatihan meskipun perpindahan parameter atau model tetap memerlukan perhatian terhadap kebocoran informasi (Kairouz dkk, 2021). Beberapa studi terkini telah mengeksplorasi penerapan FL dalam fraud detection, seperti pendekatan berbasis Graph Neural Network (Tang dan Liang, 2024) serta integrasi Explanaible Artificial Intelligence (XAI) untuk meningkatkan transparansi model (Aljunaid dkk, 2025).

Walaupun demikian, implementasi FL untuk sektor keuangan masih menghadapi tiga tantangan fundamental. Pertama, distribusi data antar institusi umumnya bersifat non-independent and identically distributed (Non-IID) karena setiap bank memiliki karakteristik nasabah, segmentasi pasar, dan profil risiko yang berbeda. Heterogenitas data ini terbukti menurunkan performa model FL terutama pada algoritma agregasi standar seperti FedAvg (Li dkk, 2021). Kedua, fraud detection juga menghadirkan permasalahan class imbalance yang ektrem dimana proporsi transaksi fraud terhadap transaksi normal seringkali kurang dari 1% (Lopez-Rojas dkk, 2016). Hal ini menjadi lebih parah dalam implementasi FL karena beberapa client mungkin memilki sangat sedikit atau bahkan tidak memiliki sampel kelas minoritas. Ketiga, sebagian besar mekanisme agregasi FL yang sudah ada dirancang khusus untuk model berbasis gradient descent, sementara model berbasis tree yang justru terbukti unggul untuk data tabular masih kurang terintegrasi dalam implementasi FL (Grinsztajn dkk, 2022).

Model berbasis tree, khususnya Extreme Gradient Boosting (XGBoost), telah menjadi pilihan utama untuk klasifikasi pada data tabular karena kemampuannya menangani missing values, fitur heterogen, dan interaksi non-linear secara efisien (Chen dan Guestrin, 2016). Karakteristik yang dimiliki model berbasis tree ini sangat relevan untuk fraud detection yang umumnya berbasis data transaksional terstruktur. Namun, implementasi XGBoost dalam FL mengalami masalah yang mendasar dimana pelatihan XGBoost konvensional dalam FL mengharuskan pertukaran gradient dan hessian per-node. Hal ini mengakibatkan risiko kebocoran privasi karena gradient dapat digunakan untuk merekonstruksi data mentah dan juga mengakibatkan frekuensi komunikasi yang sangat tinggi bergantung pada kedalaman dan jumlah tree (Zhu dkk, 2019). Sebagai solusi, Ma dkk (2023) memperkenalkan FedXGBllr (Federated XGBoost with Learnable Learning Rates), yaitu sebuah kerangka gradient-less yang mengagregasi tree ensembles antar client dan mempelajari bobot kontribusi setiap pohon melalui one-layer 1D Convolutional Neural Network. Pendekatan ini terbukti menurunkan communication overhead hingga 25-700 kali lipat sekaligus menghilangkan kebutuhan untuk berbagi gradient.

Meskipun FedXGBllr menawarkan kontribusi yang signifikan sebagai solusi dalam implementasi FL untuk fraud detection, evaluasi original FedXGBllr dilakukan pada dataset publik berskala umum seperti HIGGS, SUSY, dan a9a yang tidak memiliki karakteristik sektor keuangan. Dataset-dataset tersebut tidak memiliki class imbalance ekstrem maupun struktur fitur transaksional pada fraud detection sehingga performa FedXGBllr pada kondisi partisi Non-IID dengan rasio kelas minoritas di bawah 1% masih belum dieksplorasi. Selain itu, studi lainnya seperti Aljunaid dkk (2025) yang telah menerapkan FL untuk fraud detection dengan model konvensional seperti Logistic Regression, Support Vector Machine, dan Gradient Boosting Machine dengan agregasi best-model selection belum membandingkan dengan kerangka tree-based modern seperti FedXGBllr.

Selain model parametrik dan model berbasis tree, implementasi model deep learning pada FL juga dipertimbangkan untuk financial fraud detection. Model deep learning yang digunakan adalah FFD yang menggunakan arsitektur 1D Convolutional Neural Network dan model tabular Transformer berbasis FT-Transformer yang disebut dengan BERT (Gorishniy dkk, 2021). Penambahan kedua model tersebut memperluas perbandingan agar mencakup karakteristik model linear, model berbasis tree, dan model berbasis deep learning dalam menghadapi distribusi data yang heterogen dalam konteks financial fraud.

Dalam sektor keuangan, aspek lain yang juga menjadi perhatian adalah explainability. Model fraud detection tidak cukup hanya akurat tetapi juga harus dapat dijelaskan kepada auditor, regulator, dan juga nasabah (Bussmann dkk, 2021). Shapley Additive exPlanations (SHAP) telah menjadi standar interpretasi model ML karena dasar game theory yang konsisten (Lundberg dan Lee, 2017). Namun, penerapan SHAP pada lingkungan FL menimbulkan pertanyaan terkait stabilitas hasil interpretasi antar client ketika model dilatih di bawah kondisi data yang heterogen dan juga terkait pengaruh distribusi data antar client pada feature importance yang diperoleh. Aljunaid dkk (2025) telah mempertimbangkan XAI pada FL namun belum menganalisis secara spesifik pengaruh mekanisme agregasi yang berbeda memengaruhi stabilitas interpretasi antar client di bawah heterogenitas data.

Berdasarkan penjelasan tersebut, terdapat tiga celah penelitian yang saling terkait. Pertama, minimnya eksplorasi model berbasis tree dalam FL untuk financial fraud detection khususnya pada kerangka gradient-less seperti FedXGBllr. Kedua, belum adanya perbandingan sistematis antara model dengan agregasi gradient (LR dan SVM dengan FedAvg), model deep learning dengan agregasi accuracy-weighted FedAvg, dan model dengan agregasi tree ensemble (FedXGBllr) dalam satu kerangka eksperimen yang terkontrol. Ketiga, belum adanya analisis mengenai dampak FL pada kondisi Non-IID dan class imbalance ekstrem terhadap stabilitas interpretasi SHAP antar client.

Penelitian ini dilakukan untuk menjawab ketiga celah tersebut melalui evaluasi komparatif yang sistematis. Dataset PaySim (Lopez-Rojas dkk, 2016), yang merupakan simulasi transaksi mobile money dengan rasio fraud sekitar 0,13%, digunakan sebagai benchmark karena karakteristiknya yang merepresentasikan tantangan nyata deteksi fraud pada sektor keuangan. Selain itu, digunakan pula dataset ULB Credit Card yang bersifat riil namun dengan fitur yang dianonimkan, serta dataset Bank Account Fraud (BAF) (Jesus dkk, 2022), yaitu data sintetis yang dikembangkan berdasarkan data nyata dengan fitur bermakna sehingga memperkuat interpretabilitas. Skenario Non-IID dibentuk melalui partisi Dirichlet untuk merefleksikan heterogenitas data antar instirusi dan kerangka Flower (Beutel dkk, 2022) digunakan sebagai infrastruktur simulasi FL.

## 1.2. Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, penelitian ini merumuskan tiga pertanyaan penelitian sebagai berikut:

1. Bagaimana perbandingan performa model Federated Learning berbasis tree (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine dengan agregasi best-model selection, serta model deep learning (FFD berbasis 1D-CNN dan BERT berbasis tabular Transformer) dengan agregasi accuracy-weighted FedAvg dalam mendeteksi fraud finansial?

2. Bagaimana pengaruh heterogenitas distribusi data (IID dan Non-IID) serta penanganan class imbalance (SMOTE) terhadap performa model dalam federated learning?

3. Bagaimana karakteristik explainability model FL berbasis tree, model berbasis gradient, dan model deep learning, ditinjau dari konsistensi feature importance berbasis SHAP antar client dan stabilitasnya di bawah kondisi Non-IID?

## 1.3. Batasan Masalah

Agar penelitian ini terfokus, terarah, dan dapat dipertanggungjawabkan secara metodologis, ditetapkan sejumlah batasan masalah sebagai berikut:

1. Penelitian ini dibatasi pada deteksi financial fraud dengan paradigma pelatihan cross-silo Federated Learning (FL) yang disimulasikan dengan menggunakan Framework Flower. Validation set dan test set dipertahankan secara terpusat sebagai penyederhanaan simulasi dengan training set dipartisikan kepada lima client.

2. Model yang dievaluasi dalam penelitian ini terbatas pada enam algoritma, yaitu Logistic Regression (LR), Support Vector Machine (SVM), Gradient Boosting Machine (GBM), Financial Fraud Detection network (FFD), dan tabular Transformer (FT-Transformer, dilabeli BERT), serta FedXGBllr.

3. Skema agregasi yang digunakan dalam penelitian ini terbatas pada empat mekanisme, yaitu FedAvg untuk model parametrik (LR, SVM), best-model selection untuk GBM, accuracy-weighted FedAvg untuk model deep learning (FFD, BERT), serta tree ensemble aggregation dengan learnable learning rates untuk FedXGBllr.

4. Heterogenitas data antar client disimulasikan melalui partisi Dirichlet dengan eksperimen pelatihan utama membandingkan kondisi IID dengan Non-IID pada 𝛼 = 0,5.

5. Teknik penanganan class imbalance yang dibandingkan sebagai perlakuan eksperimen adalah SMOTE (Synthetic Minority Oversampling Technique) dan tanpa SMOTE. SMOTE diterapkan pada data pelatihan masing-masing client untuk skenario federated secara lokal serta pada training set terpusat untuk baseline. Target rasio minoritas terhadap mayoritas ditetapkan sebesar 1:100 dengan K-Nearest Neighbors k=5.

6. Analisis explainability dibatasi pada SHapley Additive exPlanations (SHAP), lebih tepatnya LinearSHAP, TreeSHAP, dan KernelSHAP. Variasi estimator KernelSHAP diperiksa melalui pengulangan seed dan pemeriksaan dilakukan secara post-hoc dalam lingkungan simulasi.

## 1.4. Tujuan

Berdasarkan rumusan masalah yang telah ditetapkan, penelitian ini memiliki tiga tujuan utama yang saling terkait, yaitu:

1. Mengevaluasi dan membandingkan performa model Federated Learning berbasis tree (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine dengan agregasi best-model selection, serta model deep learning (FFD berbasis 1D-CNN dan BERT berbasis tabular Transformer) dengan agregasi accuracy-weighted FedAvg dalam mendeteksi fraud finansial.

2. Menganalisis pengaruh heterogenitas distribusi data (IID dan Non-IID) serta penanganan class imbalance (SMOTE) terhadap performa model dalam federated learning.

3. Menganalisis karakteristik explainability model FL berbasis tree, model berbasis gradient, dan model deep learning, ditinjau dari konsistensi feature importance berbasis SHAP antar client dan stabilitasnya di bawah kondisi Non-IID.

## 1.5. Manfaat

Penelitian ini diharapkan memberikan manfaat baik secara teoritis maupun praktis sebagai berikut:

1. Memperkaya literatur Federated Learning dengan menyediakan benchmark komparatif yang mencakup empat paradigma agregasi, yaitu FedAvg untuk model parametrik, best-model selection untuk GBM, accuracy-weighted FedAvg untuk model deep learning, dan tree ensemble aggregation berbasis learnable learning rates untuk FedXGBllr dalam satu kerangka eksperimen pada tiga dataset deteksi financial fraud.

2. Memberikan bukti empiris mengenai performa FedXGBllr pada data fraud dengan class imbalance dan partisi Dirichlet Non-IID sebagai perluasan evaluasi dari dataset publik berskala umum yang digunakan pada penelitian asalnya.

3. Memberikan kontribusi awal terhadap kajian Explainable Federated Learning melalui analisis stabilitas interpretasi SHAP antar client dan antar paradigma agregasi yang masih jarang ditelaah dalam literatur.

4. Menjadi dasar metodologis bagi pengembangan kerangka evaluasi yang menggabungkan performa deteksi dan stabilitas interpretasi pada model FL parametrik, berbasis tree, dan deep learning untuk data tabular, termasuk pemeriksaan variasi estimator SHAP sebelum menarik kesimpulan mengenai perbedaan antar client.
