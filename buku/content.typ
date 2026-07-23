//=============================================================================
// ISI TUGAS AKHIR — Amadeo Yesa (5025231160)
// Ported faithfully from "Proposal TA v2.1".
// BAB 1–3 terisi dari proposal; BAB 4–5 masih stub.
//
// Catatan teknis:
//   - Penomoran persamaan mengikuti nomor bab (mis. (2.1)) secara otomatis.
//   - Gambar & tabel memakai #figure; referensi silang memakai @label sehingga
//     nomor "Gambar 2.1" / "Tabel 3.1" dihasilkan otomatis oleh template.
//   - Sitasi memakai @key (parentetis) atau #cite(<key>, form: "prose")
//     (naratif "Penulis (Tahun)").
//=============================================================================

#counter(page).update(1)
#set page(numbering: "1")

// Penomoran persamaan per-bab: (nomor-bab . nomor-persamaan)
#set math.equation(numbering: n => {
  let ch = counter(heading.where(level: 1)).get().first()
  numbering("(1.1)", ch, n)
})

// ---------------------------------------------------------------------------
// BAB 1 — PENDAHULUAN
// ---------------------------------------------------------------------------

= PENDAHULUAN

Bab ini menguraikan latar belakang, rumusan masalah, batasan masalah, tujuan,
serta manfaat dari penelitian yang dilakukan.

== Latar Belakang

Transformasi digital sektor keuangan dalam dekade terakhir telah mendorong
peningkatan volume transaksi elektronik secara eksponensial, sekaligus
memperluas permukaan serangan (attack surface) bagi praktik kecurangan
finansial (financial fraud). Laporan Association of Certified Fraud Examiners
(ACFE) menunjukkan bahwa organisasi global kehilangan rata-rata 5% dari
pendapatan tahunan mereka akibat fraud, dengan kerugian median per kasus
mencapai USD 145.000 @acfe2024. Di sisi lain, kompleksitas modus penipuan terus
berkembang seiring adopsi teknologi finansial baru, sehingga pendekatan deteksi
konvensional berbasis aturan (rule-based) menjadi semakin tidak memadai karena
keterbatasannya dalam menangkap pola anomali yang dinamis dan adaptif
@hilal2022. Sebagai respons, paradigma berbasis machine learning (ML) telah
diadopsi secara luas untuk mendeteksi pola fraud melalui analisis data historis
transaksi.

Meskipun demikian, sebagian besar implementasi ML pada deteksi fraud masih
bersifat terpusat (centralized), yaitu mengharuskan agregasi data mentah dari
berbagai institusi ke satu titik komputasi. Pendekatan ini menimbulkan
persoalan serius terkait privasi, kerahasiaan data nasabah, dan kepatuhan
terhadap regulasi seperti General Data Protection Regulation (GDPR), serta
Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi (PDP). Hal
ini secara struktural menghambat kolaborasi antar lembaga keuangan dalam
membangun model deteksi fraud yang robust, padahal pola fraud yang efektif
sering kali baru terdeteksi melalui pembelajaran lintas institusi.

Federated Learning (FL) muncul sebagai paradigma alternatif yang memungkinkan
beberapa pihak melatih model secara kolaboratif tanpa memindahkan data mentah
dari masing-masing pemiliknya @mcmahan2023fedavg. Dalam skema ini, komputasi
pelatihan dilakukan secara lokal pada setiap client, dan hanya parameter atau
pembaruan model yang dikomunikasikan ke server pusat untuk diagregasi.
Pendekatan ini telah terbukti relevan untuk sektor keuangan karena menjawab
kebutuhan privasi dan kepatuhan regulasi sekaligus tetap memungkinkan
kolaborasi antar institusi @kairouz2021. Sejumlah studi terkini telah
mengeksplorasi penerapan FL untuk deteksi fraud, mulai dari pendekatan berbasis
Graph Neural Network @tang2024 hingga integrasi Explainable Artificial
Intelligence (XAI) untuk meningkatkan transparansi model @aljunaid2025.

Walaupun demikian, implementasi FL di domain finansial masih menghadapi tiga
tantangan fundamental yang belum sepenuhnya terpecahkan. Pertama, distribusi
data antar institusi umumnya bersifat non-independent and identically
distributed (Non-IID), karena setiap bank memiliki karakteristik nasabah,
segmentasi pasar, dan profil risiko yang berbeda. Heterogenitas ini terbukti
menurunkan performa konvergensi model FL, terutama pada algoritma agregasi
standar seperti FedAvg @li2021noniid. Kedua, deteksi fraud secara inheren
menghadirkan permasalahan class imbalance yang ekstrem, di mana proporsi
transaksi fraud terhadap transaksi normal seringkali kurang dari 1%
@lopezrojas2016paysim. Hal ini diperparah pada lingkungan FL karena beberapa
client mungkin memiliki sangat sedikit atau bahkan tidak memiliki sampel kelas
minoritas. Ketiga, sebagian besar mekanisme agregasi FL yang ada dirancang
khusus untuk model berbasis gradient descent, sementara model berbasis tree
yang justru terbukti unggul untuk data tabular masih kurang terintegrasi ke
dalam ekosistem FL @grinsztajn2022.

Model berbasis tree, khususnya Extreme Gradient Boosting (XGBoost), telah lama
menjadi state-of-the-art untuk klasifikasi pada data tabular karena
kemampuannya menangani fitur heterogen, nilai missing, dan interaksi non-linear
secara efisien @chen2016xgboost. Karakteristik ini sangat relevan untuk deteksi
fraud yang umumnya berbasis data transaksional terstruktur. Namun, integrasi
XGBoost ke dalam skema FL horizontal menghadapi kendala teknis fundamental:
pelatihan XGBoost konvensional dalam FL mengharuskan pertukaran gradient dan
hessian per-node, yang berimplikasi pada (a) frekuensi komunikasi yang sangat
tinggi karena bergantung pada kedalaman dan jumlah pohon, serta (b) risiko
kebocoran privasi karena gradient dapat dieksploitasi untuk merekonstruksi data
pelatihan @zhu2019deepleakage. Sebagai solusi, #cite(<ma2023fedxgbllr>, form: "prose")
memperkenalkan FedXGBllr (Federated XGBoost with Learnable Learning Rates),
sebuah kerangka gradient-less yang mengagregasi tree ensembles antar client dan
mempelajari bobot kontribusi setiap pohon melalui one-layer 1D Convolutional
Neural Network. Pendekatan ini terbukti menurunkan communication overhead
hingga 25–700 kali lipat dibandingkan metode sebelumnya, sekaligus menghilangkan
kebutuhan untuk berbagi gradient.

Meskipun FedXGBllr menawarkan kontribusi metodologis yang signifikan, evaluasi
originalnya dilakukan pada dataset publik berskala umum seperti HIGGS, SUSY, dan
a9a yang tidak merefleksikan karakteristik domain finansial. Dataset-dataset
tersebut tidak memiliki class imbalance ekstrem maupun struktur fitur
transaksional yang khas pada deteksi fraud, sehingga performa FedXGBllr pada
kondisi yang lebih realistis khususnya partisi Non-IID berbasis distribusi
Dirichlet dengan rasio kelas minoritas di bawah 1% masih merupakan ranah yang
belum dieksplorasi. Lebih jauh lagi, studi pembanding seperti
#cite(<aljunaid2025>, form: "prose") memang telah menerapkan FL untuk deteksi
fraud dengan berbagai model konvensional (Logistic Regression, Support Vector
Machine, dan Gradient Boosting Machine) menggunakan skema agregasi best-model
selection namun belum membandingkannya dengan kerangka tree-based modern seperti
FedXGBllr secara sistematis.

Dimensi lain yang turut menjadi perhatian adalah aspek explainability. Dalam
domain keuangan, model deteksi fraud tidak cukup hanya akurat, tetapi juga harus
dapat dijelaskan kepada auditor, regulator, dan nasabah yang terdampak keputusan
@bussmann2021. SHapley Additive exPlanations (SHAP) telah menjadi standar untuk
interpretasi model ML berkat fondasi teori permainannya yang konsisten
@lundberg2017shap. Namun, penerapan SHAP pada lingkungan FL menyimpan pertanyaan
terbuka terkait stabilitas hasil interpretasi ketika model dilatih di bawah
kondisi Non-IID, dan pengaruh variasi distribusi data antar client pada feature
importance yang diperoleh. #cite(<aljunaid2025>, form: "prose") telah membuka
diskusi tentang FL berbasis XAI, namun belum mengkaji secara spesifik bagaimana
mekanisme agregasi yang berbeda antara FedAvg untuk model parametrik, best-model
selection untuk GBM, accuracy-weighted FedAvg untuk model deep learning, dan
agregasi tree ensemble pada FedXGBllr memengaruhi stabilitas interpretasi antar
client di bawah heterogenitas data.

Berdasarkan pemetaan tersebut, terdapat tiga celah penelitian (research gap)
yang saling terkait. Pertama, minimnya eksplorasi model tree-based dalam
ekosistem FL untuk deteksi fraud finansial, khususnya pada kerangka
gradient-less seperti FedXGBllr. Kedua, belum tersedianya perbandingan
sistematis antara model berbasis agregasi gradient (LR, SVM dengan FedAvg),
agregasi berbasis seleksi model (GBM dengan best-model selection), agregasi
accuracy-weighted FedAvg untuk model deep learning (FFD berbasis 1D-CNN dan BERT
berbasis tabular Transformer), dan agregasi tree ensemble (FedXGBllr) dalam satu
kerangka eksperimen yang terkontrol. Ketiga, belum adanya analisis empiris
mengenai dampak skema FL khususnya pada kondisi Non-IID dan class imbalance
ekstrem terhadap stabilitas interpretasi SHAP antar client dan antar paradigma
model.

Penelitian ini diajukan untuk menjawab ketiga celah tersebut melalui evaluasi
komparatif yang sistematis. Dataset PaySim @lopezrojas2016paysim, yang merupakan
simulasi transaksi mobile money dengan rasio fraud sekitar 0,13%, digunakan
sebagai benchmark karena karakteristiknya yang merepresentasikan tantangan nyata
deteksi fraud pada sektor keuangan. Skenario Non-IID dibangun melalui partisi
Dirichlet untuk merefleksikan heterogenitas data antar institusi, dan kerangka
Flower @beutel2022flower digunakan sebagai infrastruktur simulasi FL. Dengan
demikian, hasil penelitian ini diharapkan tidak hanya memperkaya literatur FL
berbasis tree, tetapi juga memberikan landasan praktis bagi industri perbankan,
otoritas regulator seperti Otoritas Jasa Keuangan (OJK) dan Pusat Pelaporan dan
Analisis Transaksi Keuangan (PPATK), serta pengembang sistem deteksi fraud yang
harus menyeimbangkan akurasi, privasi, dan transparansi model.

== Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, terdapat kesenjangan antara
keadaan yang ada, yaitu minimnya studi komparatif sistematis antara model
berbasis gradient, model berbasis tree, dan model deep learning dalam ekosistem
Federated Learning (FL) untuk deteksi fraud finansial, serta belum dipahaminya
dampak skema FL terhadap stabilitas interpretasi model, dengan keadaan yang ingin
dicapai, yaitu pemahaman empiris yang utuh mengenai performa dan karakteristik
explainability dari keempat paradigma agregasi FL pada kondisi yang
merepresentasikan tantangan nyata sektor keuangan, yakni class imbalance ekstrem
dan distribusi data Non-IID antar institusi. Berdasarkan kesenjangan tersebut,
penelitian ini merumuskan tiga pertanyaan penelitian sebagai berikut:

+ Bagaimana perbandingan performa model Federated Learning berbasis tree
  (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support
  Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine dengan
  agregasi best-model selection, serta model deep learning (FFD berbasis 1D-CNN
  dan BERT berbasis tabular Transformer) dengan agregasi accuracy-weighted FedAvg
  dalam mendeteksi fraud finansial?
+ Bagaimana pengaruh heterogenitas distribusi data (IID dan Non-IID) serta
  penanganan class imbalance (SMOTE) terhadap performa model dalam federated
  learning?
+ Bagaimana karakteristik explainability model FL berbasis tree, model berbasis
  gradient, dan model deep learning, ditinjau dari konsistensi feature importance
  berbasis SHAP antar client dan stabilitasnya di bawah kondisi Non-IID?

== Batasan Masalah

Agar penelitian ini terfokus, terarah, dan dapat dipertanggungjawabkan secara
metodologis, ditetapkan sejumlah batasan masalah sebagai berikut:

+ *Batasan Domain dan Dataset.* Penelitian ini dibatasi pada deteksi financial
  fraud dengan menggunakan dataset Financial Fraud Detection Dataset yang
  merupakan turunan dari simulator PaySim @lopezrojas2016paysim, bersumber dari
  repositori publik Kaggle. Dataset lain dengan domain berbeda (misalnya kartu
  kredit, insurance fraud, atau anti-money laundering) tidak digunakan agar
  pembanding antar model dilakukan pada distribusi data yang sama. Pembatasan
  ini diperlukan untuk menjaga validitas internal eksperimen serta untuk
  memastikan ketersediaan label ground truth yang konsisten di seluruh skenario
  uji.
+ *Batasan Model yang Dievaluasi.* Model yang dievaluasi dalam penelitian ini
  terbatas pada enam algoritma, yaitu Logistic Regression (LR) dan Support
  Vector Machine (SVM) sebagai representasi model parametrik dengan agregasi
  FedAvg, Gradient Boosting Machine (GBM) sebagai representasi model tree-based
  dengan agregasi best-model selection mengikuti skema
  #cite(<aljunaid2025>, form: "prose"), Financial Fraud Detection network (FFD)
  berupa 1D Convolutional Neural Network dan sebuah tabular Transformer
  (FT-Transformer, dilabeli BERT) sebagai representasi model deep learning, serta
  FedXGBllr @ma2023fedxgbllr sebagai representasi model tree-based dengan agregasi
  tree ensemble berbasis learnable learning rates. Model lain seperti Random
  Forest atau pendekatan berbasis Graph Neural Network (GNN) tidak dievaluasi
  karena di luar fokus perbandingan paradigma agregasi yang menjadi inti
  penelitian ini.
+ *Batasan Skema Agregasi Federated Learning.* Skema agregasi yang digunakan
  dalam penelitian ini terbatas pada empat mekanisme, yaitu FedAvg untuk model
  parametrik (LR, SVM), best-model selection untuk GBM, accuracy-weighted FedAvg
  untuk model deep learning (FFD, BERT), serta tree ensemble aggregation dengan
  learnable learning rates untuk FedXGBllr. Skema agregasi lanjutan seperti
  FedProx, SCAFFOLD, FedAvgM, atau FedNova tidak dievaluasi karena di luar ruang
  lingkup perbandingan dan akan memperluas dimensi eksperimen melebihi kapasitas
  penelitian ini.
+ *Batasan Skenario Heterogenitas Data.* Heterogenitas data antar client
  disimulasikan menggunakan partisi Dirichlet sebagai representasi label
  distribution skew, dengan parameter $alpha$ yang akan divariasikan untuk
  merefleksikan tingkat Non-IID yang berbeda. Skema heterogenitas lain seperti
  feature distribution skew, concept drift, atau quantity skew murni tidak
  menjadi fokus penelitian ini, meskipun akan dibahas secara konseptual sebagai
  pembatas validitas hasil. Pembatasan ini diperlukan agar dampak Non-IID dapat
  dianalisis secara terisolasi dan terkontrol.
+ *Batasan Lingkungan Eksperimen.* Penelitian ini menggunakan simulasi Federated
  Learning yang dibangun di atas kerangka kerja Flower @beutel2022flower, dengan
  implementasi mengikuti baseline hfedxgboost pada repositori resmi Flower.
  Dengan demikian, penelitian ini tidak mencakup evaluasi pada lingkungan FL
  terdistribusi secara fisik (real-world deployment), pengukuran latensi jaringan
  riil, kegagalan client yang heterogen secara perangkat keras (system
  heterogeneity), maupun aspek cross-device FL. Penelitian ini berlaku pada
  konteks cross-silo FL dengan asumsi honest-but-curious clients, sehingga
  ancaman seperti serangan Byzantine, backdoor attack, atau model poisoning
  berada di luar cakupan. Jumlah client yang disimulasikan dibatasi pada rentang
  yang umum digunakan dalam literatur cross-silo FL, yaitu antara 5 hingga 10
  client.
+ *Batasan Mekanisme Privasi Tambahan.* Penelitian ini tidak mengintegrasikan
  mekanisme privasi tambahan seperti Differential Privacy (DP), Homomorphic
  Encryption (HE), atau Secure Multi-Party Computation (SMPC). Aspek privasi yang
  dievaluasi terbatas pada properti inheren dari masing-masing skema agregasi
  khususnya karakteristik gradient-less pada FedXGBllr dan tidak menyentuh
  penambahan lapisan kriptografi atau noise injection. Pembatasan ini dimaksudkan
  agar perbandingan performa antar model tidak dipengaruhi oleh trade-off privasi
  utilitas yang akan diperkenalkan oleh mekanisme tambahan tersebut.
+ *Batasan Penanganan Class Imbalance.* Teknik penanganan class imbalance yang
  dievaluasi terbatas pada SMOTE (Synthetic Minority Oversampling Technique) yang
  diterapkan secara lokal pada setiap client sebelum proses pelatihan federated.
  Pendekatan global SMOTE tidak digunakan karena akan melanggar prinsip privasi
  FL. Teknik lain seperti undersampling, cost-sensitive learning, focal loss,
  atau Adaptive Synthetic Sampling (ADASYN) tidak dievaluasi sebagai variabel
  utama, melainkan dijadikan komponen studi ablasi (dengan dan tanpa SMOTE).
+ *Batasan Metrik Evaluasi.* Evaluasi performa model menggunakan empat metrik
  utama, yaitu Area Under the Precision-Recall Curve (AUPRC) sebagai metrik utama
  mengingat karakteristik imbalanced data, F1-score, Precision, dan Recall.
  Metrik Accuracy tidak dijadikan acuan utama karena kurang merepresentasikan
  kinerja model pada kondisi class imbalance ekstrem. Metrik tambahan seperti
  Area Under the ROC Curve (AUC-ROC) dapat dilaporkan sebagai pelengkap tetapi
  tidak menjadi dasar penarikan kesimpulan utama.
+ *Batasan Metode Explainability.* Analisis explainability dibatasi pada metode
  SHapley Additive exPlanations (SHAP) @lundberg2017shap. Metode interpretasi
  lain seperti Local Interpretable Model-agnostic Explanations (LIME), Integrated
  Gradients, atau attention-based explanation tidak digunakan agar fokus analisis
  terarah pada konsistensi dan stabilitas feature importance SHAP antar client
  dan antar paradigma model. Komputasi SHAP dilakukan secara lokal pada setiap
  client terhadap model global yang telah dilatih, kemudian diagregasi untuk
  analisis stabilitas. Analisis counterfactual explanation serta evaluasi
  kualitas interpretasi melalui user study berada di luar cakupan penelitian ini.
+ *Batasan Aspek Non-Teknis.* Aspek non-teknis seperti dampak regulasi spesifik,
  biaya implementasi industri, integrasi dengan sistem core banking, serta
  evaluasi kepuasan pengguna akhir (auditor, analis fraud) tidak dibahas dalam
  penelitian ini. Kontribusi praktis yang dirumuskan bersifat rekomendasi
  metodologis dan implikasi teoretis, bukan rekomendasi implementasi produksi.

== Tujuan

Berdasarkan rumusan masalah yang telah ditetapkan, penelitian ini memiliki tiga
tujuan utama yang saling terkait, yaitu:

+ Mengevaluasi dan membandingkan performa model Federated Learning berbasis tree
  (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support
  Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine (GBM)
  dengan agregasi best-model selection, serta model deep learning (FFD berbasis
  1D-CNN dan BERT berbasis tabular Transformer) dengan agregasi accuracy-weighted
  FedAvg dalam tugas deteksi financial fraud, dengan menggunakan metrik AUPRC,
  F1-score, Precision, dan Recall.
+ Menganalisis pengaruh kondisi distribusi data Non-IID berbasis partisi
  Dirichlet dan penerapan teknik Synthetic Minority Oversampling Technique
  (SMOTE) terhadap performa keenam model melalui studi ablasi.
+ Menganalisis karakteristik explainability keenam model menggunakan metode
  SHapley Additive exPlanations (SHAP), khususnya untuk mengkaji konsistensi
  feature importance antar client dan stabilitas interpretasi di bawah kondisi
  Non-IID, sebagai kontribusi terhadap diskursus Explainable Federated Learning
  di domain keuangan.

== Manfaat

Penelitian ini diharapkan memberikan manfaat baik secara teoritis maupun praktis
sebagai berikut:

*A. Manfaat Teoritis*

+ Memperkaya literatur Federated Learning dengan menyediakan benchmark
  komparatif sistematis pertama yang mengintegrasikan empat paradigma agregasi,
  yaitu FedAvg untuk model parametrik, best-model selection untuk GBM,
  accuracy-weighted FedAvg untuk model deep learning (FFD, BERT), dan tree
  ensemble aggregation berbasis learnable learning rates untuk FedXGBllr dalam
  satu kerangka eksperimen yang terkontrol pada konteks deteksi financial fraud.
+ Memberikan bukti empiris pertama mengenai performa FedXGBllr pada kondisi class
  imbalance ekstrem dan partisi Dirichlet Non-IID, yaitu skenario yang
  merepresentasikan tantangan nyata domain keuangan namun belum diuji pada paper
  aslinya yang hanya menggunakan dataset publik berskala umum.
+ Memberikan kontribusi awal terhadap kajian Explainable Federated Learning
  melalui analisis stabilitas interpretasi SHAP antar client dan antar paradigma
  agregasi, yang masih jarang ditelaah dalam literatur.
+ Menjadi dasar metodologis bagi pengembangan kerangka evaluasi yang
  menggabungkan dimensi performa dan explainability untuk model FL berbasis tree
  pada data tabular.

*B. Manfaat Praktis*

+ Bagi industri perbankan dan lembaga jasa keuangan, penelitian ini memberikan
  rekomendasi berbasis bukti mengenai pemilihan paradigma model FL yang sesuai
  untuk membangun sistem deteksi fraud kolaboratif lintas institusi tanpa harus
  berbagi data nasabah, sehingga dapat mendukung kepatuhan terhadap regulasi
  privasi seperti Undang-Undang Pelindungan Data Pribadi (UU PDP).
+ Bagi otoritas regulator seperti Otoritas Jasa Keuangan (OJK) dan Pusat
  Pelaporan dan Analisis Transaksi Keuangan (PPATK), hasil penelitian ini dapat
  menjadi referensi teknis dalam menyusun pedoman pemanfaatan teknologi Federated
  Learning untuk pengawasan dan deteksi transaksi mencurigakan, terutama dalam
  menjawab kebutuhan akan model yang akurat sekaligus transparan dan dapat
  diaudit.
+ Bagi pengembang sistem deteksi fraud, penelitian ini menyediakan panduan
  praktis mengenai trade-off antara performa, ketahanan terhadap heterogenitas
  data, dan interpretabilitas model dalam memilih arsitektur FL yang tepat sesuai
  konteks operasional, termasuk implikasi penggunaan SMOTE pada skema federated.
+ Bagi komunitas akademik, hasil eksperimen dan implementasi penelitian ini dapat
  menjadi baseline yang direplikasi untuk pengembangan riset lanjutan, baik dalam
  pengujian skema agregasi alternatif (FedProx, SCAFFOLD), integrasi mekanisme
  privasi tambahan (Differential Privacy), maupun ekstensi ke domain keuangan
  lainnya.

// ---------------------------------------------------------------------------
// BAB 2 — TINJAUAN PUSTAKA
// ---------------------------------------------------------------------------

= TINJAUAN PUSTAKA

Bab ini memuat hasil penelitian terdahulu yang relevan serta dasar teori yang
melandasi penelitian ini.

== Hasil Penelitian Terdahulu

Penelitian-penelitian terdahulu yang relevan dengan penelitian ini digunakan
sebagai rujukan dengan tujuan memetakan posisi penelitian ini di tengah
literatur yang ada serta mengidentifikasi celah riset yang akan diisi.
Pembahasan disusun menjadi empat kelompok tematik yang saling terkait, yaitu (1)
deteksi financial fraud berbasis machine learning dan deep learning terpusat,
(2) penerapan Federated Learning (FL) untuk deteksi fraud, (3) integrasi model
berbasis tree ke dalam kerangka FL, dan (4) penerapan Explainable Artificial
Intelligence (XAI) untuk deteksi fraud baik dalam skema terpusat maupun
federated.

=== Deteksi Financial Fraud Berbasis Machine Learning dan Deep Learning Terpusat

Penelitian deteksi fraud secara terpusat telah berkembang pesat selama satu
dekade terakhir, dengan eksplorasi luas pada arsitektur deep learning.
#cite(<sharma2022>, form: "prose") mengembangkan pendekatan credit card fraud
detection berbasis Auto-Encoder yang dipadukan dengan klasifikasi deep neural
network. Auto-encoder digunakan untuk mempelajari representasi laten dari pola
transaksi normal, sedangkan transaksi yang menunjukkan rekonstruksi dengan error
tinggi diklasifikasikan sebagai anomali. Pendekatan ini berhasil meningkatkan
kemampuan deteksi pada data dengan ketidakseimbangan kelas, namun tetap
mengasumsikan bahwa seluruh data transaksi dapat dikumpulkan pada satu titik
komputasi.

Pengembangan lebih lanjut dilakukan oleh #cite(<baghdadi2024>, form: "prose")
yang mengusulkan pendekatan ensemble learning yang menggabungkan energy-based
Restricted Boltzmann Machine (RBM) dengan extended Long Short-Term Memory
(xLSTM) untuk predictive analytics pada deteksi fraud kartu kredit. Hasil
eksperimen menunjukkan bahwa kombinasi pembelajaran representasi berbasis energi
dengan arsitektur sekuensial mampu menangkap pola temporal yang kompleks pada
transaksi finansial. Meskipun kedua penelitian tersebut menunjukkan keunggulan
teknis, keduanya beroperasi dalam paradigma terpusat sehingga tidak menjawab
persoalan privasi dan kepatuhan regulasi yang menjadi krusial dalam konteks
lintas institusi. Lebih jauh lagi, model-model berbasis deep learning tersebut
bersifat black-box, sehingga aspek interpretabilitas menjadi sulit dipenuhi
tanpa metode XAI tambahan.

=== Penerapan Federated Learning untuk Deteksi Financial Fraud

Sebagai jawaban atas keterbatasan paradigma terpusat, sejumlah peneliti mulai
mengeksplorasi penerapan Federated Learning untuk deteksi fraud finansial.
#cite(<suvarna2020>, form: "prose") merupakan salah satu kontributor awal yang
mendemonstrasikan penerapan FL pada credit card fraud detection. Penelitian
tersebut menggunakan model konvensional yang dilatih secara federated dengan
FedAvg, dan menunjukkan bahwa pendekatan FL mampu memberikan performa yang
sebanding dengan pelatihan terpusat sambil tetap menjaga privasi data pelanggan.
Namun, penelitian tersebut belum mempertimbangkan kompleksitas distribusi data
Non-IID yang merupakan kondisi nyata pada kolaborasi antar institusi keuangan.

Penelitian yang lebih lanjut oleh #cite(<venkatakrishna2024>, form: "prose")
memperluas eksplorasi tersebut dengan mengintegrasikan arsitektur deep learning
ke dalam kerangka FL untuk deteksi fraud kartu kredit. Pendekatan ini berhasil
meningkatkan akurasi deteksi melalui pemanfaatan kapasitas representasi deep
neural network, namun masih mewarisi kelemahan model deep learning dalam hal
interpretabilitas dan tetap mengandalkan agregasi berbasis FedAvg yang sensitif
terhadap heterogenitas data.

Perkembangan paling menjanjikan datang dari #cite(<tang2024>, form: "prose") yang
mengusulkan kerangka federated graph learning untuk deteksi fraud kartu kredit.
Penelitian tersebut memanfaatkan struktur graf untuk memodelkan hubungan antar
entitas transaksi, kemudian melatih Graph Neural Network (GNN) secara federated
di atas kerangka tersebut. Hasilnya menunjukkan bahwa pendekatan berbasis graf
mampu menangkap pola fraud yang bersifat relasional, yang sulit dideteksi oleh
model konvensional. Walaupun demikian, pendekatan ini menambahkan kompleksitas
komputasi yang signifikan dan tetap berbasis paradigma agregasi berbasis
gradient, sehingga belum mengeksplorasi kemungkinan penggunaan model berbasis
tree yang justru terbukti unggul untuk data tabular transaksional.

#cite(<aljunaid2025>, form: "prose") memberikan kontribusi penting dengan
mengusulkan kerangka FL berbasis XAI untuk deteksi fraud perbankan. Penelitian
tersebut menggunakan tiga model konvensional, yaitu Logistic Regression (LR),
Support Vector Machine (SVM), dan Gradient Boosting Machine (GBM), yang dilatih
secara federated dengan skema agregasi best-model selection, yaitu pemilihan
model dengan akurasi terbaik di antara seluruh client. Hasilnya menunjukkan
bahwa model GBM mencapai performa terbaik, dan integrasi SHAP berhasil
memberikan transparansi terhadap keputusan model. Meskipun pendekatan ini telah
memperkenalkan dimensi explainability ke dalam FL, penelitian tersebut belum
membandingkan hasilnya dengan model FL berbasis tree yang lebih modern seperti
FedXGBllr, dan belum mengkaji secara spesifik bagaimana skema agregasi yang
berbeda memengaruhi stabilitas interpretasi SHAP di bawah kondisi Non-IID.

=== Integrasi Model Berbasis Tree ke dalam Kerangka Federated Learning

Eksplorasi model berbasis tree dalam ekosistem FL relatif terbatas dibandingkan
model berbasis gradient. Hal ini disebabkan oleh karakteristik fundamental model
tree, khususnya XGBoost, yang membangun struktur pohon berdasarkan urutan sampel
data, sehingga skema agregasi standar seperti FedAvg tidak dapat diterapkan
secara langsung. #cite(<ma2023fedxgbllr>, form: "prose") mengusulkan FedXGBllr
(Federated XGBoost with Learnable Learning Rates), sebuah kerangka inovatif untuk
pelatihan XGBoost secara federated dalam horizontal setting yang tidak bergantung
pada pertukaran gradient dan hessian antar client. Setiap client melatih tree
ensemble secara lokal, kemudian server mengagregasi seluruh tree ensemble dan
melatih one-layer 1D Convolutional Neural Network (CNN) untuk mempelajari
learning rate setiap pohon secara global. Pendekatan ini terbukti menurunkan
communication overhead hingga 25–700 kali lipat dibandingkan metode sebelumnya
seperti SimFL, sekaligus menghilangkan risiko kebocoran privasi melalui gradient.

Walaupun demikian, evaluasi original FedXGBllr dilakukan pada dataset publik
berskala umum seperti HIGGS, SUSY, dan a9a yang tidak merefleksikan
karakteristik domain finansial. Dataset-dataset tersebut tidak memiliki class
imbalance ekstrem maupun struktur fitur transaksional yang khas pada deteksi
fraud. Selain itu, evaluasi original menggunakan partisi data yang seimbang
(equal split) dan tidak menguji performa pada skenario Non-IID berbasis
distribusi Dirichlet yang lebih realistis. Aspek explainability model juga belum
dieksplorasi dalam penelitian #cite(<ma2023fedxgbllr>, form: "prose") padahal
FedXGBllr memiliki potensi interpretasi melalui struktur pohon yang transparan
dan bobot learning rate yang dapat dipelajari.

=== Explainable AI untuk Deteksi Financial Fraud

Aspek explainability menjadi krusial dalam domain keuangan karena keputusan
model harus dapat dipertanggungjawabkan kepada auditor, regulator, dan nasabah
yang terdampak. #cite(<doshivelez2017>, form: "prose") memberikan fondasi
konseptual mengenai pentingnya interpretable machine learning dengan mengusulkan
kerangka evaluasi yang sistematis untuk mengukur kualitas interpretasi. Mereka
menekankan bahwa interpretabilitas tidak hanya bermanfaat untuk transparansi,
tetapi juga untuk identifikasi bias, validasi domain pengetahuan, serta
peningkatan kepercayaan pengguna terhadap sistem berbasis ML.

#cite(<bussmann2021>, form: "prose") mengaplikasikan kerangka tersebut pada
manajemen risiko kredit dengan memanfaatkan SHAP (SHapley Additive exPlanations)
untuk menjelaskan keputusan model XGBoost. Hasil penelitian mereka menunjukkan
bahwa SHAP mampu memberikan feature importance yang konsisten dan dapat diaudit,
sehingga cocok untuk diadopsi pada domain finansial yang ketat regulasinya.
Penelitian ini menjadi pijakan penting bahwa kombinasi model berbasis tree
dengan SHAP merupakan pasangan yang efektif untuk skenario keuangan.

#cite(<aljunaid2025>, form: "prose") telah memperkenalkan integrasi XAI ke dalam
kerangka FL untuk deteksi fraud perbankan. Namun, analisis SHAP yang mereka
lakukan masih bersifat agregat dan belum mengkaji bagaimana variasi distribusi
data antar client memengaruhi konsistensi feature importance. Pertanyaan
mengenai stabilitas interpretasi SHAP di bawah kondisi Non-IID dan
perbandingannya antar paradigma agregasi FL, antara FedAvg untuk model
parametrik, best-model selection untuk GBM, dan tree ensemble aggregation untuk
FedXGBllr, masih menjadi ranah yang belum dieksplorasi.

=== Rangkuman Penelitian Terdahulu

Berdasarkan kajian di atas, penelitian ini memposisikan diri pada irisan tiga
bidang yang belum diintegrasikan secara sistematis dalam literatur, yaitu: (1)
penerapan FL untuk deteksi financial fraud dengan class imbalance ekstrem dan
distribusi Non-IID, (2) integrasi model berbasis tree, khususnya FedXGBllr yang
bersifat gradient-less, ke dalam ekosistem FL untuk domain keuangan, dan (3)
analisis stabilitas interpretasi SHAP antar paradigma agregasi FL. @tab-2-1
merangkum hasil penelitian terdahulu yang dikaji, beserta identifikasi celah
riset yang menjadi dasar pengajuan penelitian ini.

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, 1.4fr, 2fr, 1.7fr, 2fr, 2fr),
      align: (center, left, left, left, left, left),
      table.header(
        [*No*], [*Peneliti (Tahun)*], [*Judul / Topik*], [*Metode*],
        [*Hasil Utama*], [*Gap terhadap Penelitian Ini*],
      ),
      [1], [#cite(<sharma2022>, form: "prose")],
      [Credit Card Fraud Detection Using Deep Learning Based on Auto-Encoder],
      [Auto-Encoder + DNN, paradigma terpusat],
      [Auto-encoder berhasil mempelajari representasi laten transaksi normal sehingga anomali terdeteksi via reconstruction error],
      [Paradigma terpusat sehingga tidak menjawab privasi data; model bersifat black-box tanpa analisis explainability],

      [2], [#cite(<baghdadi2024>, form: "prose")],
      [Ensemble Learning Approach Using Energy-based RBM and xLSTM],
      [Ensemble RBM + xLSTM, paradigma terpusat],
      [Kombinasi pembelajaran representasi berbasis energi dengan arsitektur sekuensial menangkap pola temporal kompleks],
      [Tidak mempertimbangkan skenario federated; tidak mengintegrasikan dimensi explainability],

      [3], [#cite(<suvarna2020>, form: "prose")],
      [Credit Card Fraud Detection Using Federated Learning Techniques],
      [FL dengan model konvensional + FedAvg],
      [Mendemonstrasikan kelayakan FL untuk deteksi fraud dengan performa sebanding pelatihan terpusat],
      [Belum mempertimbangkan distribusi Non-IID dan class imbalance ekstrem; belum mengeksplorasi model tree-based; tidak ada analisis XAI],

      [4], [#cite(<venkatakrishna2024>, form: "prose")],
      [Deep Learning-based Credit Card Fraud Detection in Federated Learning],
      [Deep Learning + FedAvg],
      [Mengintegrasikan deep learning ke dalam FL untuk meningkatkan akurasi deteksi],
      [Mewarisi keterbatasan interpretabilitas deep learning; belum membandingkan dengan model tree-based; tidak menguji robustness terhadap Non-IID],

      [5], [#cite(<tang2024>, form: "prose")],
      [Credit Card Fraud Detection Based on Federated Graph Learning],
      [Federated GNN dengan agregasi berbasis gradient],
      [Pendekatan berbasis graf efektif menangkap pola fraud relasional],
      [Kompleksitas komputasi tinggi; tetap berbasis agregasi gradient; belum mengeksplorasi model tree; tanpa analisis stabilitas XAI antar client],

      [6], [#cite(<aljunaid2025>, form: "prose")],
      [Secure and Transparent Banking: Explainable AI-Driven FL for Financial Fraud Detection],
      [LR, SVM, GBM dengan best-model selection; integrasi SHAP],
      [GBM mencapai performa terbaik di antara model konvensional; SHAP berhasil memberikan transparansi keputusan],
      [Belum membandingkan dengan FL berbasis tree modern (FedXGBllr); analisis SHAP bersifat agregat tanpa kajian stabilitas antar client di bawah Non-IID],

      [7], [#cite(<ma2023fedxgbllr>, form: "prose")],
      [Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates (FedXGBllr)],
      [Tree ensemble aggregation + learnable learning rates via 1D CNN, gradient-less],
      [Performa setara state-of-the-art dengan komunikasi 25–700× lebih efisien; menghilangkan kebutuhan berbagi gradient],
      [Evaluasi pada dataset publik umum (HIGGS, SUSY, a9a); belum diuji pada konteks fraud finansial dengan class imbalance ekstrem dan Non-IID Dirichlet; tidak ada analisis explainability],

      [8], [#cite(<doshivelez2017>, form: "prose")],
      [Towards a Rigorous Science of Interpretable Machine Learning],
      [Kerangka konseptual evaluasi interpretabilitas],
      [Mengusulkan taksonomi evaluasi interpretabilitas (application-grounded, human-grounded, functionally-grounded)],
      [Kerangka konseptual murni; menjadi fondasi untuk penelitian ini namun tidak memberikan implementasi konkret pada FL],

      [9], [#cite(<bussmann2021>, form: "prose")],
      [Explainable Machine Learning in Credit Risk Management],
      [XGBoost terpusat + SHAP pada manajemen risiko kredit],
      [SHAP konsisten dan dapat diaudit untuk model tree-based pada domain finansial],
      [Paradigma terpusat; tidak mengkaji bagaimana skema federated memengaruhi stabilitas SHAP],
    )
  ],
  caption: [Rangkuman Hasil Penelitian Terdahulu],
) <tab-2-1>

== Dasar Teori

=== Financial Fraud Detection

Financial fraud didefinisikan sebagai tindakan disengaja yang dilakukan oleh
seseorang atau kelompok untuk memperoleh keuntungan finansial secara tidak sah
melalui penipuan, manipulasi, atau penyalahgunaan kepercayaan dalam sistem
keuangan @hilal2022. Bentuk fraud yang umum meliputi penyalahgunaan kartu
kredit, money laundering, identity theft, dan transaksi mobile money yang
mencurigakan. Karakteristik utama yang membedakan deteksi fraud dari
permasalahan klasifikasi lainnya adalah ketidakseimbangan kelas yang ekstrem
(extreme class imbalance), di mana proporsi transaksi fraud terhadap transaksi
normal seringkali kurang dari 1%.

Pendekatan deteksi fraud secara umum dapat dikelompokkan menjadi tiga kategori.
Pendekatan rule-based mengandalkan aturan yang ditetapkan oleh ahli domain,
namun kurang adaptif terhadap pola fraud baru. Pendekatan machine learning
memanfaatkan algoritma pembelajaran untuk mengidentifikasi pola dari data
historis transaksi, dan pendekatan deep learning yang menggunakan arsitektur
jaringan saraf dalam untuk menangkap pola non-linear yang kompleks. Penelitian
ini berfokus pada pendekatan machine learning dan boosting tree karena
keunggulannya untuk data tabular transaksional.

=== Federated Learning

Federated Learning (FL) adalah paradigma pembelajaran mesin terdistribusi yang
memungkinkan beberapa pihak (client) melatih model bersama tanpa memindahkan
data mentah dari masing-masing pemiliknya @mcmahan2023fedavg. Pelatihan dilakukan
secara lokal di setiap client, dan hanya parameter atau pembaruan model yang
dikomunikasikan ke server pusat untuk diagregasi menjadi model global. Paradigma
ini secara langsung menjawab kebutuhan privasi data, kepatuhan regulasi, dan
keamanan informasi yang menjadi krusial pada sektor keuangan.

#figure(
  image("resources/fig-2-1-fl-architecture.png", width: 80%),
  caption: [Arsitektur umum Federated Learning. Sumber: #cite(<sabuhi2024microfl>, form: "prose")],
) <fig-2-1>

Berdasarkan distribusi data, FL diklasifikasikan menjadi tiga jenis utama
@yang2019federated: (1) Horizontal Federated Learning (HFL) ketika client
memiliki ruang fitur yang sama tetapi sampel yang berbeda, (2) Vertical Federated
Learning (VFL) ketika client memiliki ruang sampel yang sama tetapi fitur yang
berbeda, dan (3) Federated Transfer Learning (FTL) ketika ruang fitur dan ruang
sampel berbeda. Penelitian ini menggunakan paradigma HFL karena setiap institusi
keuangan diasumsikan memiliki struktur fitur transaksi yang sama tetapi sampel
nasabah yang berbeda.

Berdasarkan skala dan jumlah client, FL juga dibedakan menjadi cross-device FL
(jutaan perangkat seperti telepon genggam) dan cross-silo FL (puluhan organisasi
seperti bank atau rumah sakit). Penelitian ini berada pada konteks cross-silo FL
dengan asumsi honest-but-curious clients, yaitu client mengikuti protokol
pelatihan dengan jujur namun mungkin mencoba menyimpulkan informasi dari pesan
yang diterima. Proses pelatihan FL secara umum mengikuti empat tahapan iteratif
@kairouz2021:

+ *Inisialisasi:* Server menginisialisasi model global $w_0$ dan
  mendistribusikannya kepada seluruh client.
+ *Pelatihan Lokal:* Setiap client $k$ melatih model menggunakan data lokalnya
  $D_k$ untuk menghasilkan model lokal $w_t^k$.
+ *Agregasi:* Client mengirimkan parameter atau pembaruan model ke server, yang
  kemudian mengagregasi seluruh kontribusi menjadi model global $w_(t+1)$.
+ *Iterasi:* Model global terbaru dikirim kembali ke client untuk putaran
  berikutnya, hingga konvergensi tercapai atau jumlah putaran maksimum terpenuhi.

Adapun Federated Averaging (FedAvg) yang diperkenalkan oleh
#cite(<mcmahan2023fedavg>, form: "prose") merupakan algoritma agregasi paling
fundamental dalam FL. FedAvg melakukan rata-rata berbobot terhadap parameter
model dari seluruh client berdasarkan ukuran data lokal masing-masing,
sebagaimana dirumuskan pada @eq-fedavg:

$ w_(t+1) = sum_(k=1)^K n_k / n w_t^k $ <eq-fedavg>

dengan $w_(t+1)$ sebagai parameter global pada putaran ke-$(t+1)$, $K$ sebagai
jumlah client, $n_k$ sebagai jumlah sampel pada client ke-$k$, $n = sum n_k$
sebagai total sampel, dan $w_t^k$ sebagai parameter lokal client ke-$k$ pada
putaran ke-$t$. FedAvg cocok untuk model parametrik yang dilatih dengan gradient
descent seperti Logistic Regression dan Support Vector Machine, namun tidak dapat
diterapkan secara natural untuk model berbasis tree karena struktur pohon tidak
dapat dirata-ratakan secara element-wise.

Untuk model yang tidak kompatibel dengan agregasi berbobot seperti FedAvg,
#cite(<aljunaid2025>, form: "prose") memperkenalkan skema best-model selection
yang merumuskan agregasi sebagai pemilihan model dengan kinerja terbaik di antara
seluruh client, sebagaimana dirumuskan pada @eq-bestmodel:

$ W^* = op("arg max", limits: #true)_(W_i) A(W_i, V_i) $ <eq-bestmodel>

dengan $W^*$ sebagai bobot model global terpilih, $W_i$ sebagai bobot model dari
client ke-$i$, $A(dot)$ sebagai fungsi evaluasi kinerja (misalnya akurasi atau
AUPRC), dan $V_i$ sebagai validation set. Skema ini cocok untuk model Gradient
Boosting Machine (GBM) yang strukturnya tidak dapat dirata-ratakan, dengan
trade-off berupa penyederhanaan agregasi dan potensi kehilangan informasi dari
client yang tidak terpilih.

=== Model Parametrik (Gradient-Based)

Logistic Regression adalah model klasifikasi linear yang memodelkan probabilitas
keluaran biner menggunakan fungsi sigmoid. Model ini bekerja dengan mempelajari
sebuah vektor bobot $w$ yang merepresentasikan pengaruh setiap fitur terhadap
kemungkinan kelas tertentu. Hasil kombinasi linear antara fitur masukan dan bobot
kemudian ditransformasikan oleh fungsi sigmoid agar menghasilkan probabilitas
pada rentang 0 hingga 1.

Pelatihan LR dilakukan dengan meminimalkan binary cross-entropy loss melalui
gradient descent, sehingga modelnya kompatibel dengan agregasi FedAvg.
Keunggulan utama LR adalah interpretabilitas koefisien yang langsung menunjukkan
arah dan besarnya pengaruh setiap fitur. Namun, LR memiliki keterbatasan dalam
menangkap interaksi non-linear antar fitur, sehingga kurang optimal untuk pola
fraud yang kompleks.

Adapun Support Vector Machine (SVM) merupakan algoritma klasifikasi yang mencari
hyperplane pemisah optimal dengan margin maksimum antara dua kelas
@cortes1995svm. Secara intuitif, SVM berusaha menggambar garis pemisah yang
sejauh mungkin dari titik-titik data terdekat di kedua kelas, sehingga model
lebih robust terhadap data baru.

Pada pelatihan SVM, terdapat parameter regularisasi $C$ yang mengontrol
trade-off antara lebar margin dan toleransi terhadap kesalahan klasifikasi. Untuk
data yang tidak linearly separable, SVM dapat diperluas dengan kernel trick
(misalnya RBF atau polynomial) yang memetakan data ke ruang berdimensi lebih
tinggi. Pada konteks penelitian ini, SVM linear digunakan agar parameternya dapat
diagregasi menggunakan FedAvg.

=== Model Berbasis Tree

Gradient Boosting Machine (GBM) adalah model ensemble yang membangun pohon
keputusan secara aditif dan berurutan, dengan setiap pohon dilatih untuk
memperbaiki kesalahan pohon sebelumnya. Prediksi akhir model adalah jumlah
berbobot dari prediksi seluruh pohon dalam ensemble, dengan kontribusi setiap
pohon dikontrol oleh learning rate. Mekanisme ini memungkinkan GBM membangun
model yang kuat dari banyak weak learner berupa pohon-pohon dangkal.

GBM unggul dalam menangani fitur heterogen, interaksi non-linear, dan nilai
missing, sehingga sangat cocok untuk data tabular transaksional. Namun, struktur
pohonnya yang berupa urutan kondisi percabangan tidak dapat dirata-ratakan secara
langsung, sehingga GBM memerlukan skema agregasi alternatif seperti best-model
selection dalam konteks FL.

Extreme Gradient Boosting (XGBoost) merupakan implementasi GBM yang dioptimalkan
oleh #cite(<chen2016xgboost>, form: "prose") dengan penambahan regularisasi
eksplisit, dukungan komputasi paralel, dan penanganan missing value yang efisien.
XGBoost menambahkan komponen regularisasi pada fungsi loss-nya untuk mengontrol
kompleksitas pohon, sehingga model lebih tahan terhadap overfitting dibandingkan
GBM klasik.

Untuk membangun setiap pohon, XGBoost menggunakan informasi turunan pertama dan
turunan kedua dari fungsi loss untuk mengevaluasi kualitas setiap kandidat split
di tiap node. Pendekatan berbasis turunan inilah yang membuat XGBoost sangat
efisien, namun sekaligus menjadi tantangan utama saat hendak diintegrasikan ke
dalam FL: dalam skema federated tradisional, client harus saling bertukar
informasi turunan tersebut, yang berimplikasi pada (a) frekuensi komunikasi yang
sangat tinggi, dan (b) potensi kebocoran privasi karena informasi turunan dapat
dieksploitasi untuk merekonstruksi data pelatihan @zhu2019deepleakage.

FedXGBllr yang diusulkan oleh #cite(<ma2023fedxgbllr>, form: "prose") merupakan
kerangka horizontal federated XGBoost yang dirancang untuk mengatasi keterbatasan
XGBoost konvensional dalam FL. Inovasi utamanya terletak pada sifat
gradient-less: client tidak perlu bertukar informasi turunan apapun, melainkan
hanya mengirimkan tree ensemble yang sudah jadi.

#figure(
  image("resources/fig-2-2-fedxgbllr-architecture.jpg", width: 90%),
  caption: [Arsitektur FedXGBllr. Sumber: #cite(<ma2023fedxgbllr>, form: "prose"). (a) tahap tree ensemble aggregation dari seluruh client dan (b) struktur one-layer 1D CNN untuk mempelajari learning rate setiap pohon.],
) <fig-2-2>

Mekanismenya terdiri dari dua tahap utama. Tahap pertama, setiap client melatih
XGBoost tree ensemble lokal menggunakan datanya sendiri, kemudian mengirimkan
seluruh tree ensemble tersebut ke server. Server lalu menggabungkan seluruh tree
ensemble dari semua client menjadi satu kumpulan pohon agregat yang besar.

Tahap kedua, server tidak sekadar menggabungkan prediksi pohon-pohon tersebut
secara naif, melainkan mempelajari bobot kontribusi (learning rate) untuk setiap
pohon melalui sebuah one-layer 1D Convolutional Neural Network (CNN) yang dilatih
secara federated dengan FedAvg. Dengan kata lain, FedXGBllr membiarkan setiap
client berkontribusi melalui struktur pohonnya, namun seberapa besar pengaruh
setiap pohon terhadap prediksi akhir ditentukan secara adaptif oleh CNN tersebut.

Karakteristik kunci FedXGBllr adalah sifat gradient-less-nya, sehingga risiko
kebocoran privasi melalui informasi turunan dapat dihindari. Selain itu, jumlah
putaran komunikasi tidak bergantung pada kedalaman atau jumlah pohon, sehingga
communication overhead berkurang secara signifikan, hingga 25–700 kali lebih
efisien dibandingkan metode FL berbasis tree sebelumnya seperti SimFL.

=== Distribusi Data Non-IID dan Partisi Dirichlet

Pada lingkungan FL, data antar client umumnya bersifat non-independent and
identically distributed (Non-IID), yang berarti setiap client memiliki distribusi
data yang berbeda. Kondisi ini mencerminkan situasi nyata pada institusi
keuangan, di mana setiap bank memiliki segmentasi nasabah, profil risiko, dan
pola transaksi yang khas. Heterogenitas data antar client terbukti menurunkan
performa konvergensi model FL, terutama pada algoritma agregasi seperti FedAvg
yang mengasumsikan distribusi data relatif seragam @li2021noniid.

Untuk mensimulasikan kondisi Non-IID secara terkontrol dan dapat direplikasi,
partisi berbasis distribusi Dirichlet umum digunakan dalam literatur FL
@hsu2019. Secara intuitif, distribusi Dirichlet dengan parameter konsentrasi
$alpha$ mengontrol seberapa heterogen distribusi label antar client. Nilai
$alpha$ kecil menghasilkan distribusi yang sangat heterogen, di mana setiap
client cenderung hanya memiliki sampel dari beberapa kelas saja, sehingga
merepresentasikan kondisi Non-IID yang ekstrem. Sebaliknya, nilai $alpha$ besar
menghasilkan distribusi yang mendekati IID, di mana proporsi setiap kelas relatif
serupa antar client. Penelitian ini menggunakan beberapa nilai $alpha$ untuk
mengamati pengaruh tingkat Non-IID terhadap performa dan stabilitas interpretasi
model.

#figure(
  image("resources/fig-2-3-dirichlet-alpha.jpg", width: 90%),
  caption: [Visualisasi pengaruh parameter $alpha$ pada distribusi label antar client. Sumber: #cite(<hsu2019>, form: "prose").],
) <fig-2-3>

=== Synthetic Minority Oversampling Technique (SMOTE)

SMOTE merupakan teknik penanganan class imbalance yang menghasilkan sampel
sintetis dari kelas minoritas untuk menyeimbangkan distribusi kelas
@chawla2002smote. Berbeda dengan teknik oversampling sederhana yang hanya
menduplikasi sampel minoritas, SMOTE membuat sampel baru dengan cara melakukan
interpolasi antara satu sampel minoritas dan tetangga terdekatnya di ruang fitur.
Hasilnya adalah sampel sintetis yang berada di antara dua sampel asli, sehingga
lebih variatif dibandingkan sekadar duplikasi.

Pada penelitian ini, SMOTE diterapkan secara lokal pada setiap client sebelum
proses pelatihan federated dimulai, agar prinsip privasi FL tetap terjaga.
Penerapan SMOTE secara global akan mengharuskan agregasi data mentah ke satu
titik komputasi, yang melanggar paradigma FL. Pendekatan SMOTE lokal ini memiliki
konsekuensi: pada client dengan jumlah sampel fraud yang sangat sedikit, SMOTE
mungkin menghasilkan sampel sintetis yang kurang representatif. Hal ini menjadi
salah satu pertimbangan dalam studi ablasi pada penelitian ini.

#figure(
  image("resources/fig-2-4-smote-illustration.jpg", width: 70%),
  caption: [Ilustrasi mekanisme SMOTE. Sumber: #cite(<chawla2002smote>, form: "prose").],
) <fig-2-4>

=== Metrik Evaluasi untuk Imbalanced Classification

Pada konteks imbalanced classification, metrik accuracy tidak dapat diandalkan
karena bias terhadap kelas mayoritas. Penelitian ini menggunakan empat metrik
utama yang lebih sensitif terhadap kelas minoritas @saito2015.

*Confusion Matrix.* Matriks ini menyajikan empat komponen dasar evaluasi: True
Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN),
di mana kelas positif merepresentasikan transaksi fraud.

*Precision* mengukur proporsi prediksi positif yang benar (@eq-precision):

$ "Precision" = "TP" / ("TP" + "FP") $ <eq-precision>

*Recall* mengukur proporsi sampel positif yang berhasil dideteksi (@eq-recall):

$ "Recall" = "TP" / ("TP" + "FN") $ <eq-recall>

*F1-score* merupakan rata-rata harmonik dari Precision dan Recall (@eq-f1):

$ F_1 = 2 dot ("Precision" dot "Recall") / ("Precision" + "Recall") $ <eq-f1>

*AUPRC* (Area Under the Precision-Recall Curve) mengukur luas area di bawah kurva
Precision-Recall yang dibentuk dari berbagai threshold klasifikasi.
#cite(<saito2015>, form: "prose") menunjukkan bahwa AUPRC lebih informatif
dibandingkan AUC-ROC pada data dengan class imbalance ekstrem, karena AUC-ROC
cenderung optimistis ketika kelas negatif jauh lebih banyak dari kelas positif.
Oleh karena itu, AUPRC dipilih sebagai metrik utama dalam penelitian ini.

=== Explainable Artificial Intelligence (XAI) dan SHAP

Explainable Artificial Intelligence (XAI) merujuk pada serangkaian metode dan
teknik yang bertujuan membuat keputusan model machine learning dapat dipahami
oleh manusia @doshivelez2017. Pada domain keuangan, explainability tidak hanya
berfungsi sebagai sarana validasi teknis, tetapi juga sebagai prasyarat regulasi
dan transparansi terhadap auditor, regulator, dan nasabah.
#cite(<doshivelez2017>, form: "prose") mengusulkan tiga taksonomi evaluasi
explainability: (1) application-grounded yang melibatkan evaluasi pada aplikasi
nyata oleh praktisi domain, (2) human-grounded yang menggunakan tugas
eksperimental dengan partisipan manusia, dan (3) functionally-grounded yang
menggunakan proksi formal tanpa keterlibatan manusia. Penelitian ini berada pada
kategori functionally-grounded dengan fokus pada konsistensi dan stabilitas
feature importance berbasis SHAP.

SHAP @lundberg2017shap adalah kerangka unifikasi untuk interpretasi prediksi
model yang berlandaskan teori permainan kooperatif (Shapley values). Bagi setiap
fitur $j$ pada sampel $x$, Shapley value $phi_j$ mengukur kontribusi rata-rata
fitur tersebut terhadap prediksi model dibandingkan dengan rata-rata prediksi
baseline. Nilai Shapley dirumuskan pada @eq-shapley:

$ phi_j = sum_(S subset.eq F without {j}) (|S|! (|F| - |S| - 1)!) / (|F|!) [f_x (S union {j}) - f_x (S)] $ <eq-shapley>

dengan $F$ sebagai himpunan seluruh fitur, $S$ sebagai subset fitur tanpa fitur
$j$, dan $f_x (S)$ sebagai prediksi model dengan hanya fitur dalam $S$ yang
teramati. SHAP memenuhi tiga properti penting: local accuracy, missingness, dan
consistency, yang menjadikannya satu-satunya metode interpretasi aditif yang
konsisten dengan teori permainan @lundberg2017shap.

#figure(
  image("resources/fig-2-5-shap-summary-plot.jpg", width: 90%),
  caption: [Contoh visualisasi SHAP dalam bentuk summary plot. Sumber: #cite(<lundberg2017shap>, form: "prose").],
) <fig-2-5>

Untuk model berbasis tree seperti GBM dan XGBoost,
#cite(<lundberg2019treeshap>, form: "prose") memperkenalkan TreeSHAP, sebuah
algoritma efisien yang menghitung exact Shapley values dalam waktu polinomial
dengan memanfaatkan struktur pohon. TreeSHAP cocok untuk model penelitian ini
karena dapat memberikan interpretasi yang akurat dan dapat dihitung dengan biaya
komputasi yang wajar.

Komputasi SHAP pada praktiknya memerlukan dua himpunan data yang berbeda peran.
Yang pertama adalah background distribution, yaitu sampel referensi yang
digunakan untuk mengaproksimasi nilai harapan model $E[f(z)]$ ketika subset fitur
tertentu diasumsikan tidak teramati. Background distribution secara konseptual
merepresentasikan distribusi data "normal" yang menjadi acuan baseline
interpretasi. Yang kedua adalah explanation data, yaitu himpunan sampel yang akan
dijelaskan kontribusi fiturnya melalui Shapley values $phi_j$. Pemilihan kedua
himpunan ini memengaruhi validitas interpretasi: background yang tidak
representatif menghasilkan baseline yang bias, sedangkan explanation data yang
berbeda antar konteks evaluasi membuat hasil interpretasi sulit dibandingkan
secara langsung.

Pada konteks evaluasi stabilitas interpretasi antar client, perbandingan
magnitude SHAP secara langsung kurang tepat karena setiap client memiliki
distribusi fitur lokal yang berbeda di bawah kondisi Non-IID, sehingga rentang
nilai $|phi_j|$ tidak setara antar client. Sebagai ilustrasi, suatu client yang
memiliki proporsi transaksi fraud lebih tinggi dapat menghasilkan magnitude SHAP
yang lebih besar pada fitur tertentu dibandingkan client lain, padahal urutan
kepentingan fitur antar keduanya bisa jadi serupa. Untuk menetralkan perbedaan
skala semacam ini, literatur interpretable machine learning merekomendasikan
penggunaan metrik berbasis peringkat (rank-based metrics) yang bekerja pada
urutan kepentingan fitur, bukan pada magnitude absolutnya @doshivelez2017. Dua
metrik berbasis peringkat yang banyak digunakan adalah Spearman rank correlation
dan Jaccard similarity, yang keduanya akan menjadi metrik utama analisis
stabilitas pada penelitian ini.

Spearman rank correlation merupakan ukuran statistik yang mengevaluasi sejauh
mana dua himpunan peringkat memiliki hubungan monotonik. Berbeda dengan Pearson
correlation yang mengukur hubungan linear pada nilai aslinya, Spearman
correlation terlebih dahulu mengonversi setiap nilai menjadi peringkatnya,
kemudian menghitung korelasi antar peringkat tersebut. Konsekuensinya, Spearman
correlation tidak sensitif terhadap perbedaan skala atau transformasi monotonik
pada data, yang menjadikannya sangat sesuai untuk membandingkan feature
importance antar client dengan rentang magnitude yang berbeda.

Dalam konteks penelitian ini, Spearman correlation dihitung antara dua vektor
peringkat fitur yang dihasilkan oleh dua client berbeda. Setiap client mengurutkan
fitur berdasarkan magnitude SHAP rerata dari fitur paling berpengaruh hingga
paling tidak berpengaruh, sehingga setiap fitur memperoleh nilai peringkat.
Korelasi Spearman antara dua vektor peringkat ini dirumuskan pada @eq-spearman:

$ rho_s = 1 - (6 sum_(j=1)^d d_j^2) / (d (d^2 - 1)) $ <eq-spearman>

dengan $d_j$ sebagai selisih peringkat fitur ke-$j$ antara dua client yang
dibandingkan, dan $d$ sebagai jumlah fitur. Nilai $rho_s$ berada pada rentang
$[-1, 1]$. Nilai mendekati 1 mengindikasikan urutan kepentingan fitur yang sangat
konsisten antar client, nilai mendekati 0 mengindikasikan ketiadaan hubungan
sistematis antar peringkat, dan nilai mendekati $-1$ mengindikasikan urutan yang
berlawanan.

Spearman correlation dipilih sebagai metrik stabilitas pada penelitian ini karena
tiga alasan utama. Pertama, metrik ini menggunakan informasi peringkat penuh
(full ranking information) dari seluruh fitur, sehingga memberikan gambaran
komprehensif mengenai kesesuaian interpretasi pada semua tingkat kepentingan.
Kedua, metrik ini scale-invariant, sehingga tetap valid meskipun magnitude SHAP
berbeda antar client akibat heterogenitas data. Ketiga, Spearman correlation
memiliki interpretasi statistik yang mapan dan dapat diuji signifikansinya,
sehingga klaim stabilitas dapat dipertanggungjawabkan secara statistik.

Jaccard similarity merupakan ukuran kesamaan antara dua himpunan yang
didefinisikan sebagai rasio kardinalitas irisan terhadap kardinalitas gabungan
kedua himpunan. Pada konteks evaluasi feature importance, Jaccard similarity
diaplikasikan secara terbatas pada sejumlah $K$ fitur teratas dari setiap client,
sehingga metrik ini berfokus pada kesepakatan antar client mengenai identitas
fitur-fitur paling berpengaruh, bukan pada keseluruhan urutan. Jaccard similarity
pada top-$K$ fitur dirumuskan pada @eq-jaccard:

$ J_K = (|T_a (K) inter T_b (K)|) / (|T_a (K) union T_b (K)|) $ <eq-jaccard>

dengan $T_a (K)$ dan $T_b (K)$ masing-masing sebagai himpunan $K$ fitur teratas
pada client $a$ dan client $b$ berdasarkan magnitude SHAP rerata. Nilai $J_K$
berada pada rentang $[0, 1]$. Nilai 1 mengindikasikan kedua client memiliki
himpunan fitur teratas yang identik, sedangkan nilai 0 mengindikasikan kedua
client tidak memiliki satu pun fitur yang sama pada peringkat teratasnya.

Jaccard similarity memberikan perspektif yang berbeda dan komplementer
dibandingkan Spearman correlation. Apabila Spearman correlation mengevaluasi
konsistensi urutan kepentingan fitur secara menyeluruh, Jaccard similarity hanya
berfokus pada kesepakatan mengenai himpunan fitur teratas tanpa memperhatikan
urutan internal di antara fitur-fitur tersebut. Sebagai ilustrasi, dua client
yang sepakat bahwa lima fitur tertentu adalah yang paling penting akan memiliki
$J_5 = 1$ meskipun urutan internal kelima fitur tersebut berbeda di antara
keduanya. Hal ini relevan secara praktis bagi auditor dan regulator yang umumnya
hanya meninjau sejumlah kecil fitur teratas dalam proses validasi model, sehingga
perbedaan urutan internal di antara fitur teratas seringkali kurang penting
dibandingkan kesepakatan mengenai identitas fitur teratas itu sendiri.

Penelitian ini menggunakan Spearman rank correlation dan Jaccard similarity
secara bersamaan karena keduanya memberikan informasi yang saling melengkapi.
Spearman correlation mengevaluasi sejauh mana urutan kepentingan fitur secara
keseluruhan konsisten antar client, sedangkan Jaccard similarity mengevaluasi
sejauh mana client memiliki kesepakatan mengenai identitas fitur-fitur teratas
yang menjadi perhatian utama. Kombinasi kedua metrik memungkinkan deteksi kondisi
yang tidak dapat ditangkap oleh metrik tunggal. Sebagai contoh, dua client dapat
memiliki Spearman correlation yang tinggi namun Jaccard similarity yang rendah
apabila sebagian besar fitur memiliki peringkat yang konsisten tetapi
fitur-fitur teratasnya berbeda. Sebaliknya, dua client dapat memiliki Jaccard
similarity yang tinggi namun Spearman correlation yang rendah apabila himpunan
fitur teratasnya identik tetapi urutan keseluruhan fiturnya berbeda. Pelaporan
kedua metrik secara bersamaan memberikan gambaran yang lebih utuh mengenai
karakteristik stabilitas interpretasi model di bawah kondisi Non-IID.

Pada lingkungan FL, perhitungan SHAP menyimpan tantangan unik karena data tidak
boleh meninggalkan client. Penelitian ini menggunakan pendekatan per-client lokal
SHAP terhadap model global, di mana setiap client menghitung SHAP values secara
lokal pada data lokalnya menggunakan model global terakhir. Hasil feature
importance per-client kemudian diagregasi di server untuk analisis komparatif.
Pendekatan ini menjaga privasi data sekaligus memungkinkan analisis stabilitas
interpretasi antar client, yang menjadi salah satu kebaruan penelitian ini dalam
konteks Explainable Federated Learning.

=== Flower Framework

Flower (Friendly Federated Learning Research Framework) merupakan kerangka kerja
open-source yang dikembangkan oleh #cite(<beutel2022flower>, form: "prose") untuk
simulasi dan implementasi sistem Federated Learning. Flower menyediakan abstraksi
tingkat tinggi yang memungkinkan peneliti mengimplementasikan berbagai skema
agregasi dan model dengan kompatibilitas terhadap backend populer seperti
PyTorch, TensorFlow, dan scikit-learn. Penelitian ini menggunakan Flower sebagai
infrastruktur simulasi FL dengan implementasi mengikuti baseline hfedxgboost pada
repositori resmi Flower.

// ---------------------------------------------------------------------------
// BAB 3 — METODOLOGI
// ---------------------------------------------------------------------------

= METODOLOGI

Bab ini menguraikan metode penelitian, dataset yang digunakan, perancangan
sistem, serta implementasi perancangan sistem.

== Metode yang Digunakan

Penelitian ini menggunakan pendekatan eksperimental berbasis sistem
(system-oriented experimental research) yang mengintegrasikan seluruh tahapan
deteksi financial fraud ke dalam satu pipeline terpadu, mulai dari akuisisi data
hingga analisis explainability. Pendekatan ini dipilih karena karakteristik
penelitian yang bertujuan membandingkan beberapa konfigurasi sistem Federated
Learning (FL) secara end-to-end. Dengan demikian, kerangka metodologi disusun
mengikuti alur kerja sistem yang reproducible dan dapat dievaluasi secara
objektif pada setiap tahapnya. Pelaksanaan penelitian mengikuti alur yang
disajikan pada @fig-3-1.

#figure(
  image("resources/fig-3-1-system-architecture.jpg", height: 40%),
  caption: [Arsitektur umum sistem penelitian.],
) <fig-3-1>

Pipeline pelaksanaan penelitian terdiri dari enam tahap teknis yang berurutan.
Tahap pertama adalah data acquisition dan preprocessing, mencakup akuisisi
dataset PaySim dari Kaggle, pembersihan kolom identifier yang tidak relevan,
feature engineering untuk menghasilkan fitur turunan terkait inkonsistensi saldo
transaksi, encoding fitur kategorikal menggunakan one-hot encoding, normalisasi
skala fitur dengan StandardScaler, serta pembagian data menggunakan stratified
sampling dengan proporsi 70:15:15 untuk training set, validation set, dan test
set. Tahap kedua adalah client partitioning, di mana training set dibagi ke
beberapa client untuk simulasi lingkungan federated, baik dengan skema IID yang
membagi data secara seragam antar client maupun skema Non-IID berbasis distribusi
Dirichlet yang merepresentasikan heterogenitas data antar institusi keuangan
dengan beberapa nilai parameter konsentrasi $alpha$. Tahap ketiga adalah local
training dan imbalance handling, di mana setiap client melatih model lokal pada
data partisinya, didahului penerapan SMOTE secara lokal untuk menyeimbangkan
kelas minoritas tanpa melanggar prinsip privasi FL. Tahap keempat adalah
federated aggregation, di mana parameter atau struktur model dari setiap client
diagregasi di server pusat menggunakan empat skema agregasi yang berbeda, yaitu
FedAvg untuk model parametrik (LR, SVM), best-model selection untuk GBM,
accuracy-weighted FedAvg untuk model deep learning (FFD, BERT), dan tree ensemble
aggregation dengan learnable learning rates untuk FedXGBllr. Tahap
kelima adalah experimental scenarios, yang mencakup empat skenario berjenjang:
centralized baseline sebagai upper-bound performa, federated IID untuk mengukur
overhead paradigma federated, federated Non-IID untuk menguji robustness terhadap
heterogenitas data, dan studi ablasi dengan dan tanpa SMOTE untuk mengisolasi
kontribusi teknik penanganan class imbalance. Tahap keenam adalah evaluation dan
explainability analysis, yang mengukur performa model menggunakan metrik
klasifikasi standar dan menganalisis konsistensi serta stabilitas interpretasi
model menggunakan SHAP secara per-client.

== Dataset yang Digunakan

=== Karakteristik Dataset

Penelitian ini menggunakan Financial Fraud Detection Dataset yang dipublikasikan
pada platform Kaggle oleh Sriharsha Eedala. Dataset tersebut merupakan turunan
dari simulator PaySim @lopezrojas2016paysim, yaitu simulator transaksi mobile
money yang dikembangkan berdasarkan log transaksi nyata dari sebuah perusahaan
jasa keuangan di Afrika. Dataset ini dipilih karena memenuhi tiga kriteria yang
relevan dengan konteks penelitian, yaitu (1) class imbalance yang ekstrem dengan
rasio fraud sekitar 0,13%, (2) struktur fitur transaksional tabular yang mewakili
karakteristik nyata sektor keuangan, dan (3) skala data yang memadai untuk
simulasi federated dengan beberapa client. Karakteristik utama dataset disajikan
pada @tab-3-1.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Kaggle (Sriharsha Eedala)],
    [Jenis data], [Transaksi mobile money tabular],
    [Jumlah baris], [± 6.362.620 transaksi],
    [Jumlah fitur], [11 kolom],
    [Label target], [isFraud (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 0,13% (kelas minoritas ekstrem)],
    [Tipe fitur], [Numerik dan kategorikal],
  ),
  caption: [Karakteristik Dataset],
) <tab-3-1>

Deskripsi setiap fitur dataset disajikan pada @tab-3-2.

#figure(
  kind: table,
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.header([*Nama Fitur*], [*Tipe*], [*Deskripsi*]),
    [step], [Numerik], [Unit waktu transaksi],
    [type], [Kategorikal], [Jenis transaksi: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER],
    [amount], [Numerik], [Nominal transaksi dalam mata uang lokal],
    [nameOrig], [Kategorikal], [Identitas pengirim],
    [oldbalanceOrg], [Numerik], [Saldo pengirim sebelum transaksi],
    [newbalanceOrig], [Numerik], [Saldo pengirim setelah transaksi],
    [nameDest], [Kategorikal], [Identitas penerima],
    [oldbalanceDest], [Numerik], [Saldo penerima sebelum transaksi],
    [newbalanceDest], [Numerik], [Saldo penerima setelah transaksi],
    [isFraud], [Biner], [Label target (1 = fraud, 0 = normal)],
    [isFlaggedFraud], [Biner], [Flag deteksi rule-based legacy],
  ),
  caption: [Deskripsi Fitur Dataset],
) <tab-3-2>

=== Pembagian dan Penggunaan Dataset

Pembagian dataset dirancang dalam dua tingkat (two-level split) untuk
merefleksikan konteks penelitian yang melibatkan baseline terpusat sekaligus
simulasi federated learning. Tingkat pertama merupakan pembagian global yang
dilakukan satu kali pada keseluruhan dataset PaySim, sedangkan tingkat kedua
merupakan partisi training set ke seluruh client untuk skenario federated.

Pada tingkat pertama, dataset dibagi menjadi tiga subset dengan proporsi
70:15:15 untuk training set, validation set, dan test set. Pembagian dilakukan
menggunakan stratified sampling berdasarkan label isFraud agar proporsi kelas
minoritas yang ekstrem (±0,13%) tetap terjaga di setiap subset. Random seed
ditetapkan secara tetap untuk menjamin reproduksibilitas. Ketiga subset memiliki
peran yang berbeda dalam pipeline penelitian. Training set (70%) menjadi sumber
data pelatihan dan akan dipartisi lebih lanjut ke seluruh client pada skenario
federated. Validation set (15%) dipertahankan secara terpusat di server simulasi
dan digunakan untuk dua keperluan, yaitu (1) sebagai dasar pemilihan model pada
skema best-model selection untuk Gradient Boosting Machine, sebagaimana
dirumuskan pada @eq-bestmodel, dan (2) sebagai dasar pemantauan konvergensi serta
keputusan early stopping selama proses pelatihan. Test set (15%) juga
dipertahankan secara terpusat di server simulasi dan hanya digunakan satu kali
pada akhir eksperimen untuk pelaporan performa final keenam model dengan metrik
AUPRC, F1-score, Precision, dan Recall. Subset ini tidak diakses selama proses
pelatihan maupun pemilihan hyperparameter untuk menghindari test set leakage.

Pada tingkat kedua, hanya training set (70%) yang dipartisi ke seluruh client,
sedangkan validation set dan test set tetap utuh di server simulasi. Partisi
dilakukan menggunakan skema IID atau Dirichlet Non-IID sehingga setiap client $k$
memperoleh subset lokal $D_k$. Di dalam setiap client, subset lokal $D_k$
digunakan secara langsung sebagai data pelatihan model lokal tanpa pemecahan
tambahan menjadi local validation, mengingat fungsi validasi telah diakomodasi
oleh validation set terpusat. Penanganan class imbalance melalui SMOTE
diaplikasikan pada $D_k$ secara lokal sebelum proses pelatihan dimulai.

Penggunaan validation set dan test set secara terpusat merupakan penyederhanaan
simulasi (simulation simplification) yang umum digunakan dalam riset federated
learning berbasis kerangka Flower @beutel2022flower untuk memungkinkan
perbandingan performa antar paradigma agregasi pada distribusi evaluasi yang
konsisten. Pada penerapan federated learning di lingkungan produksi, evaluasi
sebaiknya dilakukan secara terdistribusi melalui mekanisme federated evaluation.

== Perancangan Sistem

Sistem yang dirancang dalam penelitian ini terdiri dari empat lapisan logis yang
saling terkait, yaitu (1) lapisan data dan preprocessing, (2) lapisan partisi dan
client orchestration, (3) lapisan pelatihan model dengan empat paradigma agregasi
FL, dan (4) lapisan evaluasi yang mencakup pengukuran performa dan analisis
explainability. Arsitektur umum sistem disajikan pada @fig-3-2.

#figure(
  image("resources/fig-3-2-logical-layers.png", height: 42%),
  caption: [Lapisan logis sistem penelitian.],
) <fig-3-2>

=== Perancangan Tahap Preprocessing

Tahap preprocessing dirancang untuk mempersiapkan data tabular PaySim agar
memenuhi kebutuhan pelatihan keenam model yang digunakan dalam penelitian ini.
Rangkaian operasi preprocessing disusun secara sistematis dengan
mempertimbangkan karakteristik data transaksi keuangan serta sensitivitas
masing-masing algoritma terhadap distribusi dan skala fitur.

Langkah pertama adalah pembersihan kolom identifier, yaitu menghapus kolom
nameOrig, nameDest, dan isFlaggedFraud. Ketiga kolom tersebut merupakan
identifier unik atau legacy flag yang tidak memiliki relevansi prediktif terhadap
pemodelan deteksi fraud.
Selanjutnya dilakukan feature engineering dengan membentuk fitur turunan berupa
errorBalanceOrig yang didefinisikan sebagai newbalanceOrig − oldbalanceOrg +
amount, serta errorBalanceDest yang didefinisikan sebagai oldbalanceDest + amount
− newbalanceDest. Kedua fitur ini dirancang untuk menangkap inkonsistensi saldo
antar akun yang dalam literatur sebelumnya terbukti menjadi indikator kuat
aktivitas fraud.

Setelah proses feature engineering, dilakukan encoding terhadap fitur kategorikal
type menggunakan teknik one-hot encoding yang menghasilkan lima kolom biner
berdasarkan jenis transaksi. Tahap berikutnya adalah normalisasi skala fitur
numerik menggunakan StandardScaler untuk memastikan konsistensi rentang nilai
antar fitur. Meskipun model berbasis pohon seperti GBM dan FedXGBllr tidak
mensyaratkan proses scaling, prosedur ini tetap diterapkan demi menjaga
konsistensi pipeline lintas model. Sebagai langkah akhir, dataset dibagi menjadi
tiga subset dengan proporsi 70:15:15 untuk training, validation, dan testing
menggunakan stratified sampling berdasarkan label isFraud, sehingga distribusi
kelas minoritas tetap representatif pada setiap subset.

=== Perancangan Skema Partisi Client

Skema partisi data antar client dirancang dalam dua mode untuk merepresentasikan
kondisi distribusi data yang berbeda pada lingkungan federated learning. Kedua
mode ini ditetapkan untuk mengevaluasi performa model dalam skenario yang ideal
maupun yang lebih realistis terhadap karakteristik institusi keuangan.

Mode pertama adalah mode Independent and Identically Distributed (IID), di mana
training set terpusat dibagi secara seragam ke seluruh client dengan proporsi
label yang relatif sama. Mode ini berfungsi sebagai baseline untuk mengukur
overhead yang ditimbulkan oleh paradigma federated learning dibandingkan
pendekatan terpusat. Mode kedua adalah mode Non-IID dengan partisi berbasis
distribusi Dirichlet, di mana proporsi kelas pada setiap client ditarik dari
distribusi $p_k tilde.op "Dir"(alpha)$. Dalam penelitian ini, tiga nilai
parameter konsentrasi $alpha$ diuji untuk merepresentasikan tingkat heterogenitas
yang berbeda, yaitu $alpha = 0.5$ untuk kondisi Non-IID kuat, $alpha = 1.0$ untuk
kondisi Non-IID sedang, dan $alpha = 5.0$ untuk kondisi Non-IID ringan yang
mendekati distribusi IID. Jumlah client yang disimulasikan ditetapkan sebanyak
$K = 5$, mengacu pada konfigurasi cross-silo yang umum digunakan dalam literatur
federated learning untuk merepresentasikan kolaborasi antar institusi keuangan
berskala menengah.

#figure(
  image("resources/fig-3-3-two-level-split.jpg", width: 85%),
  caption: [Skema pembagian data dua tingkat.],
) <fig-3-3>

Skema ini menjamin bahwa validation set dan test set tetap konsisten di seluruh
skenario eksperimen, baik pada baseline terpusat, federated IID, maupun federated
Non-IID dengan berbagai nilai $alpha$. Dengan demikian, perbandingan performa
antar paradigma agregasi dilakukan pada distribusi evaluasi yang identik,
sehingga perbedaan metrik yang teramati dapat diatribusikan secara murni pada
perbedaan paradigma agregasi dan kondisi heterogenitas data, bukan pada perbedaan
distribusi evaluasi.

=== Perancangan Skema Class Imbalance Handling

SMOTE diterapkan secara lokal pada setiap client sebelum proses pelatihan
dimulai. Pemilihan SMOTE lokal, bukan SMOTE global, dilakukan untuk menjaga
prinsip privasi FL, yaitu data tidak boleh meninggalkan client. Rasio
penyeimbangan ditetapkan melalui parameter sampling_strategy sebesar 0,01, yaitu
menargetkan proporsi kelas minoritas terhadap kelas mayoritas sebesar 1:100 pada
setiap client yang memenuhi syarat. Target parsial ini dipilih, alih-alih
penyeimbangan penuh 1:1, agar sampel sintetis tidak mendominasi partisi lokal
yang berukuran besar sekaligus menjaga biaya komputasi tetap wajar. Pada client
yang memiliki sampel fraud sangat sedikit (kurang dari k_neighbors + 1 = 6),
SMOTE akan di-skip dan client tersebut tetap dilatih dengan data aslinya untuk
menghindari sintesis sampel yang tidak representatif. Client yang proporsi
fraud-nya sudah mencapai atau melampaui target juga tidak dioversample, karena
SMOTE hanya menambah sampel minoritas. Studi ablasi dengan dan tanpa SMOTE
dilakukan untuk mengisolasi pengaruh teknik ini.

=== Perancangan Pelatihan Model dan Skema Agregasi

Pelatihan model LR dan SVM dilakukan dengan gradient descent lokal pada setiap
client selama $E$ local epochs, kemudian parameter dikirim ke server untuk
diagregasi dengan FedAvg. Putaran ini diulang sebanyak $R$ global rounds hingga
konvergensi. Untuk GBM, setiap client melatih model GBM lokal lengkap, kemudian
server memilih satu model dengan AUPRC tertinggi pada validation set sebagai
model global. Model deep learning FFD dan BERT dilatih secara lokal selama $E$
local epochs, lalu parameter jaringannya diagregasi menggunakan accuracy-weighted
FedAvg, yaitu rata-rata berbobot ganda menurut ukuran data lokal sekaligus AUPRC
lokal masing-masing client @yang2019federated. Untuk FedXGBllr, proses mengikuti
dua tahap: (1) tahap tree ensemble aggregation di putaran ke-0, dan (2) tahap
pelatihan 1D CNN secara federated dengan FedAvg pada putaran 1 hingga $R$. Keenam
model dilatih dengan empat skema agregasi sebagaimana diringkas pada @tab-3-3.

#figure(
  kind: table,
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.header([*Model*], [*Kategori*], [*Skema Agregasi*]),
    [Logistic Regression (LR)], [Parametrik], [FedAvg],
    [Support Vector Machine (SVM)], [Parametrik (linear)], [FedAvg],
    [Gradient Boosting Machine (GBM)], [Tree ensemble (histogram-based)], [Best-Model Selection],
    [FFD], [Deep learning (1D-CNN)], [Accuracy-Weighted FedAvg],
    [BERT (FT-Transformer)], [Deep learning (Transformer tabular)], [Accuracy-Weighted FedAvg],
    [FedXGBllr], [Tree ensemble + CNN], [Tree Ensemble Aggregation + Learnable LR],
  ),
  caption: [Pemetaan Model dengan Skema Agregasi Federated Learning],
) <tab-3-3>

=== Perancangan Modul Evaluasi

Pengukuran performa dilakukan pada test set terpusat untuk seluruh skenario
eksperimen. Empat metrik dihitung: AUPRC (utama), F1-score, Precision, dan
Recall. Setiap eksperimen dijalankan sebanyak tiga kali dengan random seed yang
berbeda, dan hasil akhir dilaporkan sebagai rata-rata dengan deviasi standar.

Analisis explainability dirancang untuk mengukur dua dimensi yang saling
melengkapi, yaitu konsistensi interpretasi sebagai indikator kesepakatan antar
client mengenai fitur-fitur paling berpengaruh, dan stabilitas interpretasi
sebagai indikator ketahanan penjelasan model terhadap kondisi heterogenitas data.
Pengukuran kedua dimensi ini memerlukan perancangan yang eksplisit terhadap tiga
elemen, yaitu background distribution, explanation data, dan metrik agregasi
antar client.

Background distribution dirancang menggunakan sub-sample sebanyak 100 sampel dari
training data lokal masing-masing client. Ukuran ini dipilih untuk menjaga
konsistensi biaya komputasi antar client tanpa mengorbankan representativitas
baseline. Pendekatan ini menjaga prinsip privasi Federated Learning karena
background distribution tidak meninggalkan client, sekaligus memberikan baseline
yang merefleksikan distribusi lokal yang sebenarnya dipelajari oleh model.

Explanation data dirancang menggunakan subset tetap dari test set terpusat dengan
ukuran 500 sampel yang identik untuk seluruh client. Pemilihan test set terpusat
sebagai explanation data dilakukan dengan dua pertimbangan utama. Pertama,
identitas explanation data antar client menjamin bahwa perbedaan interpretasi
yang teramati murni berasal dari perbedaan model lokal terhadap model global,
bukan dari perbedaan distribusi data lokal yang dapat menjadi confounding
variable dalam analisis stabilitas. Kedua, pendekatan ini analog dengan praktik
standar pada literatur SHAP terpusat oleh #cite(<bussmann2021>, form: "prose")
sehingga hasil interpretasi tetap dapat dibandingkan dengan baseline terpusat
dalam skenario centralized. Subset explanation data dipilih secara proporsional
terhadap distribusi kelas asli, sehingga mencakup sampel transaksi normal maupun
fraud secara representatif.

Komputasi SHAP dilakukan menggunakan tiga varian explainer yang dipilih
berdasarkan karakteristik masing-masing model. Varian TreeSHAP oleh
#cite(<lundberg2019treeshap>, form: "prose") diaplikasikan pada GBM dan FedXGBllr
karena efisiensi komputasinya yang bersifat polinomial pada model berbasis pohon
serta kemampuannya menghasilkan exact Shapley values untuk struktur ensemble.
Varian LinearSHAP diaplikasikan pada Logistic Regression karena memberikan exact
Shapley values untuk model linear dengan biaya komputasi rendah. Varian
KernelSHAP diaplikasikan pada Support Vector Machine sebagai pendekatan
model-agnostik, dengan ukuran background dibatasi pada 100 sampel untuk menjaga
efisiensi komputasi mengingat kompleksitas eksponensial KernelSHAP terhadap
jumlah fitur.

Pada setiap client, komputasi SHAP menghasilkan vektor feature importance lokal
yang diperoleh melalui rerata absolut SHAP values pada seluruh sampel explanation
data. Vektor ini merefleksikan seberapa besar pengaruh setiap fitur terhadap
prediksi model global menurut perspektif client yang bersangkutan. Vektor feature
importance dari seluruh client kemudian dihimpun di server simulasi untuk
dianalisis lebih lanjut menggunakan tiga jenis komputasi statistik yang saling
melengkapi.

Komputasi pertama adalah rerata feature importance antar client yang berfungsi
sebagai indikator consensus interpretasi global. Hasil rerata ini menunjukkan
fitur-fitur yang secara umum dianggap penting oleh seluruh client, terlepas dari
heterogenitas data lokal masing-masing. Komputasi kedua adalah Spearman rank
correlation rerata antar seluruh pasangan client, yang mengukur kesesuaian urutan
kepentingan fitur antar client. Spearman rank correlation dipilih sebagai metrik
stabilitas karena bekerja pada ranah peringkat, bukan magnitude, sehingga tetap
valid meskipun rentang nilai SHAP berbeda antar client akibat distribusi fitur
lokal yang berbeda. Nilai Spearman yang mendekati 1 mengindikasikan urutan
kepentingan fitur yang sangat konsisten antar client, sedangkan nilai yang
mendekati 0 mengindikasikan interpretasi yang tidak stabil.

Komputasi ketiga adalah Jaccard similarity rerata pada lima fitur teratas (top-5)
antar seluruh pasangan client. Metrik ini mengukur proporsi fitur penting yang
sama-sama muncul pada peringkat lima teratas di dua client yang dibandingkan.
Pemilihan top-5 didasarkan pada pertimbangan praktis bahwa auditor dan regulator
umumnya hanya meninjau sejumlah kecil fitur teratas dalam proses validasi model,
sehingga kesepakatan antar client mengenai identitas fitur paling berpengaruh
menjadi indikator yang relevan secara operasional.

Kombinasi ketiga metrik ini memberikan gambaran komplementer mengenai
karakteristik explainability model. Rerata feature importance menunjukkan apa
yang diinterpretasikan sebagai penting, sedangkan Spearman rank correlation dan
Jaccard similarity menunjukkan seberapa stabil interpretasi tersebut antar client
di bawah heterogenitas data. Ketiga metrik dilaporkan untuk setiap kombinasi
model, skenario partisi, dan penerapan SMOTE. Stabilitas yang rendah pada
skenario Non-IID dengan parameter Dirichlet $alpha$ yang kecil akan
diinterpretasikan sebagai indikasi sensitivitas model terhadap heterogenitas
distribusi data antar client, yang menjadi salah satu kontribusi orisinal
penelitian ini terhadap diskursus Explainable Federated Learning.

== Implementasi Perancangan Sistem

Subbab ini menjelaskan realisasi konkret dari setiap komponen Perancangan Sistem
yang telah diuraikan pada Subbab Perancangan Sistem. Setiap subbab implementasi
disusun sejajar dengan subbab perancangan untuk memudahkan pelacakan kesesuaian
antara desain dan realisasi. Sebelum implementasi diuraikan, terlebih dahulu
disajikan spesifikasi lingkungan pengembangan yang menjadi prerequisite seluruh
modul implementasi. Implementasi penelitian ini dilakukan dalam lingkungan
komputasi yang spesifikasinya disajikan pada @tab-3-4.

#figure(
  kind: table,
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Komponen*], [*Spesifikasi*]),
    [Bahasa pemrograman], [Python 3.10.x],
    [Framework FL], [Flower (flwr) 1.5.0 dengan simulasi berbasis Ray],
    [Deep learning], [PyTorch 2.8.0, torchmetrics 1.8.2],
    [Orkestrasi konfigurasi], [Hydra 1.3.2 (FedXGBllr); YAML + argparse (model lain)],
    [Library ML klasik], [scikit-learn 1.5.0, XGBoost 2.0.0],
    [Penanganan class imbalance], [imbalanced-learn (SMOTE, ADASYN)],
    [Library XAI], [SHAP (shap)],
    [Library numerik & DataFrame], [NumPy, Pandas, SciPy],
    [Library visualisasi], [Matplotlib, Seaborn],
    [Pelacakan eksperimen], [Weights & Biases (wandb 0.15.12)],
    [Version control], [Git + GitHub],
  ),
  caption: [Spesifikasi Lingkungan Pengembangan],
) <tab-3-4>

=== Implementasi Tahap Preprocessing

Realisasi enam langkah preprocessing yang dirancang pada Subbab Perancangan Tahap
Preprocessing diorganisasi sebagai sebuah pipeline terstruktur yang
mengintegrasikan seluruh transformasi data ke dalam satu kerangka eksekusi yang
konsisten dan dapat direproduksi. Pipeline tersebut dilatih (fitted) secara
eksklusif pada training set untuk mencegah terjadinya kebocoran informasi
statistik (data leakage) dari validation set maupun test set, kemudian
diaplikasikan secara identik pada kedua himpunan tersebut.

Pipeline tersebut terdiri dari empat komponen transformasi yang dieksekusi secara
berurutan. Komponen pertama berfungsi menghapus kolom-kolom identifier yang tidak
relevan untuk pemodelan, yaitu identitas pengirim, identitas penerima, dan flag
deteksi rule-based peninggalan sistem terdahulu. Komponen kedua melakukan feature
engineering untuk menghasilkan dua fitur turunan yang merepresentasikan
inkonsistensi saldo pada akun pengirim dan akun penerima, yang secara empiris
terbukti menjadi indikator kuat aktivitas fraud. Komponen ketiga melakukan
transformasi fitur kategorikal jenis transaksi menjadi representasi biner melalui
one-hot encoding. Komponen keempat melakukan normalisasi skala fitur numerik
menggunakan transformasi standar agar setiap fitur memiliki rerata nol dan
deviasi standar satu.

Pemisahan dataset menjadi tiga himpunan dengan proporsi 70:15:15 dilakukan
menggunakan teknik stratified sampling yang mempertahankan proporsi kelas fraud
secara konsisten di seluruh himpunan, dengan random seed yang ditetapkan secara
tetap untuk menjamin reproduksibilitas eksperimen.

=== Implementasi Modul Partisi Client

Realisasi skema partisi yang dirancang pada Subbab Perancangan Skema Partisi
Client dibangun melalui dua mekanisme partisi yang berbeda. Mekanisme pertama
merepresentasikan skenario IID melalui pembagian data secara seragam ke seluruh
client dengan distribusi label yang proporsional. Mekanisme ini berfungsi sebagai
baseline untuk mengukur overhead murni paradigma federated yang terisolasi dari
faktor heterogenitas data.

Mekanisme kedua mengimplementasikan partisi Non-IID berbasis distribusi
Dirichlet. Untuk setiap kelas pada dataset, vektor proporsi yang menentukan
distribusi sampel kelas tersebut ke seluruh client ditarik dari distribusi
Dirichlet dengan parameter konsentrasi $alpha$. Nilai $alpha$ yang diuji dalam
penelitian ini ditetapkan pada tiga tingkat heterogenitas, yaitu $alpha = 0.5$
untuk merepresentasikan kondisi Non-IID yang kuat, $alpha = 1.0$ untuk Non-IID
sedang, dan $alpha = 5.0$ untuk kondisi yang mendekati IID. Reproduksibilitas
kedua mekanisme partisi dijamin melalui penetapan random seed yang konsisten,
sehingga partisi yang dihasilkan dapat direplikasi secara identik pada setiap
eksekusi eksperimen.

Penting untuk ditegaskan bahwa kedua mekanisme partisi tersebut diaplikasikan
secara eksklusif pada training set, sesuai dengan skema dua tingkat pada Subbab
Pembagian dan Penggunaan Dataset. Validation set dan test set yang dihasilkan
dari pembagian global pada Level 1 tidak mengalami partisi pada Level 2 dan tetap
dipertahankan secara utuh di server simulasi. Konsekuensinya, setiap client $k$
hanya menerima subset lokal $D_k subset D_"train"$ dan tidak memiliki akses
terhadap validation set maupun test set. Setelah partisi selesai, subset lokal
$D_k$ langsung diteruskan ke modul SMOTE lokal tanpa pemecahan tambahan, karena
fungsi validasi pada penelitian ini diakomodasi oleh validation set terpusat di
server simulasi.

=== Implementasi Modul SMOTE Lokal

Realisasi skema SMOTE lokal yang dirancang pada Subbab Perancangan Skema Class
Imbalance Handling dibangun dengan strategi pemanggilan terdistribusi pada
masing-masing client, bukan pada data terpusat. Setiap client menerima partisi
data lokalnya dan melakukan proses oversampling terhadap kelas minoritas secara
independen sebelum data hasil oversampling digunakan dalam pelatihan model lokal.
Pendekatan ini secara fundamental menjaga prinsip privasi Federated Learning
karena tidak ada pertukaran data mentah antar client maupun ke server pusat.

Mengingat distribusi sampel fraud yang tidak merata antar client pada skenario
Non-IID, modul SMOTE dilengkapi dengan mekanisme pengaman berupa pemeriksaan
kelayakan jumlah sampel kelas minoritas pada setiap client. Apabila jumlah sampel
kelas fraud pada suatu client tidak mencukupi untuk operasi interpolasi
K-Nearest Neighbors (kurang dari k_neighbors + 1 = 6 sampel), proses SMOTE pada
client tersebut diabaikan dan pelatihan dilakukan menggunakan data asli untuk
menghindari sintesis sampel yang tidak representatif. Konfigurasi rasio
penyeimbangan ditetapkan melalui parameter sampling_strategy sebesar 0,01
(target proporsi minoritas terhadap mayoritas 1:100) pada setiap client yang
memenuhi syarat; client yang telah mencapai atau melampaui target tersebut
dilewati karena SMOTE hanya menambah sampel kelas minoritas. Sebagai bagian
integral dari studi
ablasi, modul SMOTE dapat dinonaktifkan secara global melalui parameter
konfigurasi yang relevan, sehingga memungkinkan pengamatan kontribusi murni
teknik penanganan class imbalance terhadap performa akhir model.

=== Implementasi Pelatihan Model dan Skema Agregasi

Realisasi pelatihan keenam model dengan empat skema agregasi yang dirangkum pada
@tab-3-3 dibangun di atas kerangka kerja Flower, dengan setiap skema agregasi
diimplementasikan sebagai strategi terkustomisasi yang mewarisi antarmuka
strategi standar dari Flower. Pendekatan modular ini memungkinkan pertukaran
skema agregasi tanpa memodifikasi komponen lain pada pipeline sistem.

*FedAvg untuk LR dan SVM.* Skema agregasi FedAvg sebagaimana dirumuskan pada
@eq-fedavg diaplikasikan untuk model Logistic Regression dan Support Vector
Machine linear. Logistic Regression direalisasikan menggunakan implementasi
sklearn, sedangkan SVM linear direalisasikan menggunakan SGDClassifier dengan
loss hinge agar pembaruan parameternya bersifat inkremental dan dapat diakumulasi
antar putaran oleh FedAvg. Setiap client melakukan pelatihan model lokal selama
beberapa local epochs, kemudian mentransmisikan parameter model, berupa vektor
koefisien dan bias, ke server pusat untuk diagregasi melalui rata-rata berbobot
sesuai proporsi ukuran data lokal masing-masing client. Proses ini diulang secara
iteratif sebanyak $R$ global rounds hingga konvergensi tercapai.

*Best-Model Selection untuk GBM.* Skema agregasi alternatif yang dirumuskan pada
@eq-bestmodel diimplementasikan sebagai strategi terkustomisasi yang mengganti
mekanisme rata-rata berbobot dengan seleksi model berbasis kinerja. Pada setiap
putaran komunikasi, seluruh client melatih model Gradient Boosting Machine
(direalisasikan dengan HistGradientBoostingClassifier berbasis histogram demi
skalabilitas pada volume data PaySim) secara lengkap pada data lokalnya, kemudian
server pusat melakukan evaluasi seluruh model kandidat pada validation set
terpusat dan menetapkan model dengan nilai AUPRC tertinggi sebagai model global
untuk putaran selanjutnya. Pendekatan ini diadopsi karena struktur GBM yang
berbasis pohon keputusan tidak dapat dirata-ratakan secara element-wise
sebagaimana parameter numerik. Implementasi ini mengikuti formulasi yang
diusulkan oleh #cite(<aljunaid2025>, form: "prose").

*Accuracy-Weighted FedAvg untuk FFD dan BERT.* Kedua model deep learning, yaitu
FFD yang berupa 1D Convolutional Neural Network dan BERT yang berupa tabular
Transformer (FT-Transformer), diagregasi menggunakan varian FedAvg berbobot ganda
mengikuti gagasan #cite(<yang2019federated>, form: "prose"). Pada skema ini, bobot
kontribusi setiap client tidak hanya ditentukan oleh proporsi ukuran data lokal,
tetapi juga dikalikan dengan AUPRC lokal client tersebut, sehingga client dengan
data lebih banyak sekaligus performa lokal lebih baik memberikan pengaruh lebih
besar terhadap model global. Pada putaran awal ketika seluruh AUPRC lokal masih
bernilai nol, skema ini otomatis kembali ke FedAvg standar berbasis proporsi data
agar agregasi tetap terdefinisi.

*Tree Ensemble Aggregation dengan Learnable Learning Rates untuk FedXGBllr.*
Realisasi FedXGBllr mengikuti kerangka baseline yang dipublikasikan pada
repositori resmi Flower dan dijalankan dalam dua tahap. Tahap pertama
mengimplementasikan agregasi tree ensemble sebagai strategi terkustomisasi yang
menghimpun tree ensemble lokal dari setiap client, masing-masing terdiri atas 50
pohon yang dilatih menggunakan algoritma XGBoost, untuk kemudian disusun menjadi
aggregated tree ensemble berukuran $M times K$ yang merepresentasikan keragaman
model dari seluruh client. Tahap kedua merealisasikan komponen learnable learning
rates menggunakan arsitektur one-layer 1D Convolutional Neural Network, dengan
ukuran kernel dan stride yang disesuaikan terhadap jumlah pohon per-client.
Komponen CNN ini dilatih secara federated menggunakan skema FedAvg standar selama
$R$ putaran komunikasi.

Pada seluruh paradigma agregasi yang diimplementasikan, validation set terpusat
berperan sebagai sumber sinyal evaluasi global pada setiap putaran komunikasi.
Pada paradigma FedAvg untuk LR dan SVM, validation set digunakan untuk memantau
konvergensi model global dan menjadi dasar penghentian dini apabila AUPRC tidak
meningkat selama sejumlah putaran berturut-turut. Pada paradigma best-model
selection untuk GBM, validation set berfungsi sebagai dasar evaluasi seluruh
model kandidat dari setiap client untuk menentukan model dengan AUPRC tertinggi
sebagai model global, sebagaimana dirumuskan pada @eq-bestmodel. Pada paradigma
accuracy-weighted FedAvg untuk FFD dan BERT, validation set digunakan untuk
memantau konvergensi model global dan menjadi dasar penghentian dini. Pada
paradigma tree ensemble aggregation untuk FedXGBllr, validation set digunakan
untuk memantau konvergensi pelatihan komponen 1D CNN selama tahap kedua.
Penggunaan
validation set terpusat di seluruh paradigma menjamin konsistensi sinyal evaluasi
antar skema agregasi, sehingga perbedaan performa yang teramati dapat dianalisis
secara terisolasi pada level paradigma agregasi.

Konfigurasi hyperparameter yang digunakan pada seluruh model disajikan pada
@tab-3-5. Pemilihan nilai hyperparameter dilakukan secara terbatas, baik melalui
grid search sederhana maupun adopsi nilai default yang direkomendasikan oleh
literatur, untuk menjaga fokus penelitian pada perbandingan paradigma agregasi
dan menghindari potensi bias akibat optimasi hyperparameter yang ekstensif.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (1.3fr, 1fr),
      align: (left, left),
      table.header([*Parameter*], [*Nilai*]),
      table.cell(colspan: 2)[_Umum (seluruh model)_],
      [Jumlah client (K)], [5],
      [Global rounds (R)], [20],
      [Dirichlet $alpha$], [{0,5 ; 1,0 ; 5,0}],
      [Random seed], [42],
      [SMOTE: k_neighbors], [5],
      [SMOTE: sampling_strategy], [0,01 (target minoritas:mayoritas 1:100)],

      table.cell(colspan: 2)[_Logistic Regression (LR)_],
      [Local epochs (E)], [1],
      [C], [1,0],
      [max_iter], [1000],

      table.cell(colspan: 2)[_Support Vector Machine (SVM, SGDClassifier)_],
      [Local epochs (E) / max_iter per putaran], [5],
      [alpha (regularisasi L2)], [0,0001],
      [loss], [hinge (linear SVM)],

      table.cell(colspan: 2)[_Gradient Boosting Machine (GBM, HistGBM)_],
      [max_iter (n_estimators)], [100],
      [learning_rate], [0,1],
      [max_depth], [6],

      table.cell(colspan: 2)[_FFD (1D-CNN)_],
      [Local epochs (E)], [5],
      [batch_size], [80],
      [learning rate], [0,01],

      table.cell(colspan: 2)[_BERT (FT-Transformer)_],
      [Local epochs (E)], [3],
      [d_model / nhead / num_layers], [64 / 4 / 2],
      [dim_feedforward / dropout], [256 / 0,1],
      [batch_size / learning rate / weight_decay], [64 / 0,001 / 0,0001],

      table.cell(colspan: 2)[_FedXGBllr_],
      [Jumlah pohon per-client], [50],
      [XGBoost: max_depth / learning_rate], [6 / 0,1],
      [XGBoost: subsample / alpha / gamma], [0,8 / 5 / 5],
      [1D CNN kernel_size (= stride)], [sama dengan jumlah pohon per-client (50)],
      [1D CNN learning rate], [0,0005],
      [Iterasi CNN per putaran], [50],
    )
  ],
  caption: [Konfigurasi Hyperparameter Eksperimen],
) <tab-3-5>

=== Implementasi Modul Evaluasi dan SHAP

Realisasi modul evaluasi yang dirancang pada Subbab Perancangan Modul Evaluasi
mencakup dua komponen yang saling melengkapi, yaitu pengukuran performa model dan
analisis explainability, yang keduanya dieksekusi pada model global akhir setiap
skenario eksperimen.

Komponen analisis explainability merealisasikan kerangka pengukuran yang telah
dirancang pada Subbab Perancangan Modul Evaluasi dengan mengacu pada konfigurasi
data dan varian explainer yang telah ditetapkan. Implementasi dilakukan
menggunakan pustaka SHAP versi terbaru yang mendukung seluruh varian explainer
yang dibutuhkan, serta diintegrasikan dengan pipeline evaluasi sehingga komputasi
explainability dapat dijalankan secara otomatis setelah pelatihan model global
selesai.

Konfigurasi data SHAP diimplementasikan melalui dua mekanisme sampling yang
berbeda. Background distribution untuk setiap client diperoleh melalui random
sampling tanpa pengembalian dari training data lokal pasca-SMOTE dengan ukuran
100 sampel, menggunakan random seed yang ditetapkan secara konsisten antar
eksperimen untuk menjamin reproduksibilitas. Explanation data diperoleh melalui
random sampling tanpa pengembalian sebanyak 500 sampel dari test set terpusat,
dengan komposisi proporsional terhadap distribusi kelas asli sehingga mencakup
sampel transaksi normal maupun fraud. Subset explanation data yang sama digunakan
oleh seluruh client pada seluruh skenario eksperimen untuk menjamin komparabilitas
hasil interpretasi antar client dan antar paradigma agregasi.

Penting untuk dicatat bahwa pemberian akses test set yang sama kepada seluruh
client untuk komputasi SHAP bersifat sebagai penyederhanaan simulasi yang sejalan
dengan keberadaan test set terpusat di server simulasi sebagaimana dijelaskan
pada Subbab Pembagian dan Penggunaan Dataset. Pada penerapan di lingkungan
produksi, mekanisme federated SHAP computation perlu dipertimbangkan untuk
menjaga kepatuhan terhadap prinsip privasi Federated Learning. Pembatasan ini
telah dibahas pada Subbab Batasan Masalah sebagai salah satu batasan validitas
eksternal penelitian.

Implementasi explainer mengikuti pemetaan yang telah ditetapkan pada Subbab
Perancangan Modul Evaluasi. TreeSHAP diaplikasikan pada GBM dan FedXGBllr melalui
antarmuka TreeExplainer dari pustaka SHAP. Untuk FedXGBllr, komputasi dilakukan
pada aggregated tree ensemble hasil tahap pertama, dengan kontribusi setiap pohon
dibobot oleh learnable learning rates yang dipelajari oleh komponen 1D CNN pada
tahap kedua. LinearSHAP diaplikasikan pada Logistic Regression melalui
LinearExplainer, dan KernelSHAP diaplikasikan pada SVM linear melalui
KernelExplainer. Khusus untuk KernelSHAP, jumlah evaluasi fungsi dibatasi pada
nilai default pustaka untuk menjaga efisiensi komputasi mengingat kompleksitasnya
yang bersifat eksponensial terhadap jumlah fitur.

Komputasi feature importance dijalankan secara independen pada setiap client
setelah model global akhir tersedia. Setiap client menghitung SHAP values untuk
seluruh sampel explanation data, kemudian merangkumnya menjadi vektor feature
importance lokal melalui rerata absolut. Vektor-vektor ini kemudian dikirimkan ke
server simulasi untuk dihimpun menjadi matriks feature importance antar client,
yang menjadi dasar seluruh analisis komparatif berikutnya.

Pada server simulasi, dilakukan tiga jenis komputasi statistik sesuai rancangan
pada Subbab Perancangan Modul Evaluasi. Pertama, rerata feature importance antar
client dihitung sebagai indikator consensus interpretasi dan divisualisasikan
dalam bentuk bar chart untuk memberikan gambaran fitur-fitur yang secara umum
dianggap penting oleh seluruh client. Kedua, Spearman rank correlation rerata
dihitung antar seluruh pasangan client sebagai metrik stabilitas urutan
kepentingan fitur. Ketiga, Jaccard similarity rerata pada lima fitur teratas
dihitung antar seluruh pasangan client sebagai metrik stabilitas himpunan fitur
paling berpengaruh.

Hasil pengukuran ketiga metrik dilaporkan untuk setiap kombinasi model, skenario
partisi, dan penerapan SMOTE, kemudian disajikan dalam bentuk tabel komparatif
yang memungkinkan analisis lintas paradigma agregasi. Visualisasi pelengkap
berupa heatmap matriks feature importance antar client, scatter plot korelasi
peringkat antar pasangan client, serta summary plot SHAP untuk setiap model
digunakan untuk mendukung interpretasi kualitatif. Stabilitas yang menurun seiring
penurunan parameter Dirichlet $alpha$ akan diinterpretasikan sebagai indikasi
sensitivitas model terhadap heterogenitas distribusi data antar client, yang
menjadi salah satu kontribusi orisinal penelitian ini terhadap diskursus
Explainable Federated Learning.

// ---------------------------------------------------------------------------
// BAB 4 — HASIL DAN PEMBAHASAN  (stub)
// ---------------------------------------------------------------------------

= HASIL DAN PEMBAHASAN

// TODO: BAB 4 belum ada di Proposal TA v2.1. Isi dengan hasil eksperimen,
// analisis performa (AUPRC/F1/Precision/Recall), studi ablasi SMOTE, dan
// analisis stabilitas SHAP (Spearman & Jaccard) setelah eksperimen selesai.

// ---------------------------------------------------------------------------
// BAB 5 — PENUTUP  (stub)
// ---------------------------------------------------------------------------

= PENUTUP

// TODO: BAB 5 belum ada di Proposal TA v2.1. Isi dengan Kesimpulan (menjawab
// ketiga rumusan masalah) dan Saran untuk penelitian lanjutan.

//=============================================================================
// REFERENSI SINTAKS (contoh dikomentari — jangan dihapus, bukan bagian isi)
// Satu contoh tiap konstruksi template: figure, table, equation, citation.
//=============================================================================

// --- Contoh GAMBAR (figure image) ---
// #figure(
//   image("resources/fig-2-1-fl-architecture.png", width: 80%),
//   caption: [Keterangan gambar di sini.],
// ) <contoh-gambar>
// Referensi silang: @contoh-gambar  -> otomatis "Gambar 2.1"

// --- Contoh TABEL (figure kind: table; caption otomatis di ATAS) ---
// #figure(
//   kind: table,
//   table(
//     columns: (auto, 1fr),
//     table.header([*Kolom A*], [*Kolom B*]),
//     [baris 1], [nilai 1],
//     [baris 2], [nilai 2],
//   ),
//   caption: [Keterangan tabel di sini.],
// ) <contoh-tabel>
// Referensi silang: @contoh-tabel  -> otomatis "Tabel 2.1"

// --- Contoh PERSAMAAN (bernomor otomatis per-bab) ---
// $ y = m x + c $ <contoh-persamaan>
// Referensi silang: @contoh-persamaan  -> otomatis "(2.1)"

// --- Contoh SITASI ---
// Parentetis:  @mcmahan2023fedavg          -> "(McMahan dkk., 2023)"
// Naratif:     #cite(<ma2023fedxgbllr>, form: "prose")  -> "Ma dkk. (2023)"
