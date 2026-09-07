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
deteksi fraud pada sektor keuangan. Untuk menguji generalisasi lintas domain,
digunakan pula dataset ULB Credit Card yang bersifat riil namun fiturnya
teranonimkan, serta dataset Bank Account Fraud (BAF) @jesus2022baf yang
menyediakan fitur riil bernama dan bermakna semantik sehingga memperkuat analisis
interpretabilitas. Skenario Non-IID dibangun melalui partisi
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
  fraud dengan menggunakan tiga dataset publik dari repositori Kaggle, yaitu
  Financial Fraud Detection Dataset yang merupakan turunan dari simulator PaySim
  @lopezrojas2016paysim sebagai dataset utama, ULB Credit Card Fraud Detection
  Dataset sebagai dataset pembanding lintas domain, dan Bank Account Fraud (BAF)
  @jesus2022baf terbitan Feedzai (NeurIPS 2022), yang dipilih karena menyediakan
  fitur riil bernama dan bermakna semantik sehingga analisis interpretabilitas
  berbasis SHAP dapat dimaknai secara lebih substantif. Varian yang digunakan
  dari BAF adalah varian Base. Domain fraud lain seperti insurance fraud atau anti-money
  laundering tidak digunakan. Pembatasan ini diperlukan untuk menjaga validitas
  internal eksperimen serta untuk memastikan ketersediaan label ground truth yang
  konsisten di seluruh skenario uji.
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
  atau Adaptive Synthetic Sampling (ADASYN) @he2008adasyn tidak dievaluasi
  sebagai variabel utama, melainkan dijadikan komponen studi ablasi (dengan dan
  tanpa SMOTE).
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

Terdapat pula kesenjangan antara dua untai literatur mengenai penanganan class
imbalance. Untai terapan pada deteksi fraud berbasis FL — termasuk
#cite(<aljunaid2025>, form: "prose") sebagai pembanding langsung penelitian ini —
lazimnya menerapkan penyeimbangan seperti SMOTE sebagai langkah praproses baku
tanpa mengevaluasi apakah koreksi tersebut memang diperlukan; sepanjang yang
dilaporkan, ablasi eksplisit dengan dan tanpa koreksi imbalance tidak menjadi
bagian dari evaluasinya. Sebaliknya, untai metodologis menemukan bahwa koreksi
sering kali tidak diperlukan atau bahkan merugikan: #cite(<goorbergh2022harm>, form: "prose")
menunjukkan koreksi tidak meningkatkan diskriminasi namun menyebabkan overestimasi
probabilitas, dan @blagus2013smote menemukan SMOTE kurang efektif dibanding
undersampling pada data berdimensi tinggi. Penelitian ini menguji konvensi
tersebut secara empiris lintas tiga dataset dan empat paradigma agregasi,
alih-alih mengasumsikannya — sebuah pengujian terhadap konvensi yang berlaku di
literatur terapan, bukan klaim mengenai praktik terbaik.

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
titik komputasi, yang melanggar paradigma FL.

#figure(
  image("resources/fig-2-4-smote-illustration.jpg", width: 70%),
  caption: [Ilustrasi mekanisme SMOTE. Sumber: #cite(<chawla2002smote>, form: "prose").],
) <fig-2-4>

==== Keterbatasan Teoretis SMOTE pada Regime Sampel Minoritas Kecil

Efektivitas SMOTE tidak seragam di seluruh kondisi data, dan hal ini menjadi
sangat relevan pada FL dengan partisi Non-IID yang dapat menghasilkan client
bersampel fraud sangat sedikit. #cite(<weiss2004rarity>, form: "prose")
membedakan dua bentuk kelangkaan: *relative rarity*, yaitu rasio kelas yang
timpang, dan *absolute rarity*, yaitu jumlah sampel minoritas yang terlalu
sedikit secara absolut. Oversampling adalah satu-satunya keluarga teknik yang
secara langsung menangani absolute rarity — dengan menduplikasi contoh langka
(yang menurutnya tidak selalu dianjurkan), mensintesis contoh baru, atau idealnya
memperoleh contoh nyata yang benar-benar baru — sedangkan untuk relative rarity,
sampling hanya menyeimbangkan distribusi. Implikasinya penting untuk penelitian
ini: dataset secara utuh menunjukkan relative rarity, tetapi partisi Dirichlet
menginduksi *absolute rarity* pada client yang starved, dan interpolasi tidak
dapat menyembuhkan absolute rarity karena tidak menambahkan informasi baru apa
pun tentang manifold minoritas — ia hanya menata ulang informasi dari segelintir
titik nyata yang sudah ada.

Ketergantungan kualitas SMOTE pada jumlah sampel juga telah dianalisis secara
formal. @elreedy2019smote menurunkan ekspektasi dan kovariansi data hasil SMOTE
dan mengidentifikasi dimensi input, nilai K, serta jumlah sampel minoritas
sebagai faktor penentu, dengan performa pasca-SMOTE membaik seiring meningkatnya
jumlah minoritas nyata. @elreedy2024smote menunjukkan bahwa titik sintetis
cenderung terletak lebih ke dalam (inward) relatif terhadap seed-nya, sehingga
distribusi sintetis menjadi lebih terkontraksi dibanding distribusi sebenarnya,
dan menyatakan bahwa SMOTE dapat menempatkan sampel di dalam wilayah kelas
mayoritas serta memperkuat seed yang noisy.

Pada data berdimensi tinggi, degradasi ini makin nyata. @blagus2013smote
menemukan bahwa SMOTE gagal meredam bias ke arah kelas mayoritas untuk sebagian
besar klasifikator pada data berdimensi tinggi dan kurang efektif dibandingkan
random undersampling, sekaligus menurunkan variabilitas dan menginduksi korelasi
antar sampel yang dihasilkan; simulasi mereka dijalankan pada jumlah minoritas
sekecil 5 dan 10 — persis regime yang muncul pada client starved dalam penelitian
ini.

Terakhir, apakah koreksi imbalance memang diperlukan sama sekali masih
diperdebatkan. #cite(<goorbergh2022harm>, form: "prose") membandingkan tanpa
koreksi, random undersampling, random oversampling, dan SMOTE pada regresi
logistik dan ridge lintas berbagai fraksi kejadian termasuk 1%, dan menemukan
bahwa koreksi tidak meningkatkan diskriminasi namun menghasilkan overestimasi
sistematis pada probabilitas prediksi, dengan perolehan pada metrik klasifikasi
yang sebenarnya dapat dicapai hanya dengan menggeser ambang keputusan.
Kesimpulan mereka — bahwa ketimpangan kelas bukanlah masalah inheren dan koreksi
justru dapat memperburuk performa — memotivasi penelitian ini untuk memperlakukan
konfigurasi tanpa koreksi sebagai baseline dan SMOTE sebagai intervensi yang
diuji, bukan sebaliknya.

Sejumlah varian dirancang untuk memitigasi kelemahan ini — SMOTE yang membatasi
seed pada wilayah perbatasan (@han2005borderline, @bunkhumpornpat2009safelevel)
dan hibrida yang membersihkan sampel pasca-oversampling (@batista2004balancing),
sebagaimana dirangkum oleh @fernandez2018smote — namun keseluruhannya berada di
luar cakupan penelitian ini sesuai Batasan Masalah, yang membatasi penanganan
imbalance pada SMOTE standar.

==== Interpolasi pada Fitur Kategorikal (Keterbatasan SMOTE-NC)

Perlu ditegaskan bahwa penelitian ini menerapkan SMOTE standar pada matriks
fitur yang memuat kolom kategorikal hasil one-hot encoding (26 dari 55 kolom pada
BAF, 5 dari sekitar 15 pada PaySim; ULB tidak terpengaruh karena seluruh fiturnya
merupakan komponen PCA). Karena SMOTE melakukan interpolasi kontinu, sampel
sintetis memperoleh nilai pecahan pada kolom one-hot, sehingga menghasilkan
kombinasi kategori yang mustahil pada data nyata — sebuah transaksi BAF sintetis
dapat, misalnya, secara pecahan menjadi dua jenis payment_type sekaligus.
#cite(<chawla2002smote>, form: "prose") sebenarnya memperkenalkan *SMOTE-NC*
(Nominal-Continuous) pada makalah yang sama untuk data campuran nominal–kontinu,
yang menangani fitur nominal melalui voting mayoritas di antara tetangga alih-alih
interpolasi, dan pustaka `imbalanced-learn` menyediakan implementasinya melalui
`SMOTENC`. SMOTE standar tetap dipertahankan demi konsistensi dengan fokus utama
penelitian, yaitu perbandingan antar paradigma agregasi, dan migrasi ke SMOTE-NC
direkomendasikan untuk penelitian lanjutan. Konsekuensinya berbeda antar keluarga
model: model berbasis pohon dapat sebagian memulihkan indikator biner melalui
split di sekitar 0,5, sedangkan model parametrik dan deep learning mengonsumsi
nilai pecahan tersebut secara langsung.

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

Perlu ditegaskan bahwa nilai AUPRC dari sebuah pengklasifikasi acak sama dengan
prevalensi kelas positif, sehingga batas bawah AUPRC berbeda antar dataset dan
nilai AUPRC tidak dapat dibandingkan secara langsung lintas dataset. Sebagai
contoh, garis dasar acak bernilai sekitar 0,0013 untuk PaySim, 0,0017 untuk ULB,
dan 0,011 untuk BAF, sehingga skor pada BAF akan tampak lebih tinggi hanya karena
prevalensinya lebih besar. Untuk itu, penelitian ini melaporkan garis dasar acak
(prevalensi) berdampingan dengan setiap nilai AUPRC, alih-alih menormalkannya,
agar nilai AUPRC mentah tetap dapat dibandingkan di dalam satu dataset sekaligus
dibaca relatif terhadap batas bawahnya saat dibandingkan antar dataset.

*Recall\@5%FPR.* AUPRC meringkas kualitas peringkat pada seluruh ambang dan —
sebagaimana ditegaskan di atas — batas bawahnya bergerak mengikuti prevalensi,
sehingga perbandingan lintas dataset menjadi kurang langsung. Sebagai pelengkap
yang konkret secara operasional, penelitian ini juga melaporkan *Recall\@5%FPR*,
yaitu proporsi fraud yang terdeteksi ketika ambang keputusan ditetapkan agar false
positive rate (FPR) pada transaksi sah sama dengan 5% (@eq-fpr). Metrik ini tetap
diturunkan dari ambang, tetapi dibaca pada satu titik operasi tetap alih-alih pada
ambang yang disetel bebas.

$ "FPR" = "FP" / ("FP" + "TN") $ <eq-fpr>

Recall\@FPR menjawab pertanyaan yang berbeda dari AUPRC dan langsung relevan bagi
praktik: pada anggaran false alarm yang benar-benar sanggup ditinjau sebuah
institusi, berapa proporsi fraud yang tertangkap? Sebuah tim analis yang hanya
mampu meninjau 5% transaksi yang ditandai peduli pada angka ini, bukan pada luas
area di bawah sebuah kurva. Berbeda dari AUPRC, batas bawah Recall\@FPR tidak
bergantung prevalensi sehingga perbandingan antar dataset tidak memerlukan
normalisasi. Titik operasi pada FPR tetap semacam ini merupakan konvensi pelaporan
yang lazim pada deteksi fraud maupun intrusi.

Ambang yang dipilih dilaporkan pada skala skor masing-masing model, dan skala ini
tidak seragam. Untuk LR, GBM, FFD, BERT, dan FedXGBllr ambang berupa probabilitas
dalam rentang [0, 1], tetapi untuk SVM ambang berupa margin `decision_function`
bertanda (loss hinge, tanpa probabilitas). Karena itu ambang SVM bermagnitudo besar
— misalnya −128,08 pada ULB dan −49,14 pada BAF — adalah margin yang wajar, bukan
galat, dan tidak sebanding dengan ambang berskala probabilitas seperti GBM yang
berada dalam rentang [0, 1] (misalnya 0,038 pada BAF). Pembacaan naif yang
menyandingkan kedua skala tanpa memperhatikan perbedaan ini akan keliru
menyimpulkan adanya kerusakan. Selain itu, karena distribusi skor bersifat diskret
dan ambang diambil pada nilai terbesar yang masih memenuhi FPR $lt.eq$ 5%, FPR yang
benar-benar dicapai umumnya sedikit di bawah 5%; capaian ini dilaporkan berdampingan
dengan targetnya agar titik operasi dapat diverifikasi. Pada sebagian kecil sel FPR
aktual jatuh jauh di bawah target — misalnya GBM pada ULB skema IID tanpa SMOTE
hanya mencapai 0,0068 — yaitu ketika ties skor atau model yang terpangkas
menghasilkan distribusi skor kasar tanpa ambang mana pun yang dekat dengan 5%.

*Kalibrasi.* Keempat metrik di atas hanya mengukur diskriminasi dan tidak dapat
mendeteksi pergeseran skala probabilitas. Karena
#cite(<goorbergh2022harm>, form: "prose") menemukan bahwa koreksi imbalance
menyebabkan overestimasi sistematis probabilitas prediksi, penelitian ini
menambahkan metrik kalibrasi pada test set terpusat. Yang pertama adalah *Brier
score*, yaitu rerata kuadrat selisih antara probabilitas prediksi dan luaran
biner (semakin kecil semakin baik). Yang kedua adalah *calibration slope* dan
*calibration intercept*, yang mengikuti hierarki kalibrasi
#cite(<vancalster2016hierarchy>, form: "prose") dilaporkan sebagai dua angka yang
independen. *Calibration slope* adalah koefisien $b$ dari regresi logistik gabungan
tanpa penalti $y ~ a + b dot "logit"(p)$; slope 1 menandakan kalibrasi sempurna,
slope kurang dari 1 menandakan probabilitas yang terlalu ekstrem (over-confident),
sedangkan slope lebih dari 1 menandakan probabilitas yang terlalu terkompresi
(under-confident). *Calibration intercept* dilaporkan sebagai
*calibration-in-the-large*, yaitu intercept $a$ dari model offset dengan slope
dikunci pada 1, $y ~ 1 + "offset"("logit"(p))$, yang menyelesaikan
$"mean"(sigma("logit"(p) + a)) = "mean"(y)$. Angka ini dipisahkan dari slope karena
intercept pada regresi gabungan terkopel dengan slope sehingga tidak dapat
ditafsirkan sendiri — pada model yang sangat under-dispersed intercept gabungan
membengkak ke nilai yang tak bermakna, sementara calibration-in-the-large tetap
terinterpretasi. Mengikuti konvensi #cite(<goorbergh2022harm>, form: "prose"),
intercept 0 berarti terkalibrasi secara rata-rata; intercept negatif menandakan
overestimasi sistematis (probabilitas terlalu tinggi), sedangkan intercept positif
menandakan underestimasi sistematis.

Ketersediaan probabilitas berbeda antar model dan ditangani secara jujur. LR,
GBM, FFD, dan BERT mengeluarkan probabilitas terkalibrasi sehingga kalibrasi
dihitung langsung. FedXGBllr merutekan luarannya melalui 1D CNN yang lapisan
akhirnya adalah fungsi Sigmoid, sehingga keluarannya sudah berupa probabilitas
dan kalibrasi dihitung tanpa transformasi tambahan. SVM diimplementasikan sebagai
`SGDClassifier` dengan loss hinge yang hanya mengekspos `decision_function`,
yaitu margin dan bukan probabilitas; untuk SVM metrik kalibrasi dilaporkan sebagai
`NA`, karena melewatkan margin hinge melalui sigmoid akan menghasilkan kalibrasi
yang direkayasa, bukan yang terukur.

*Ambang klasifikasi.* AUPRC bersifat bebas ambang (threshold-independent) sehingga
tidak terpengaruh pergeseran skala probabilitas, sedangkan F1, Precision, dan
Recall bergantung pada ambang. Karena oversampling menggeser skala probabilitas,
ambang tetap (misalnya 0,5) akan membuat ketiga metrik itu bergerak karena alasan
yang tidak terkait kualitas peringkat. Untuk menetralkan konfound ini, ambang
keputusan disetel per-arm pada validation set terpusat dengan memaksimalkan F1,
lalu diterapkan tanpa perubahan pada test set; kebijakan ini seragam untuk seluruh
model dan skenario. #cite(<elor2022smote>, form: "prose") — sebuah preprint —
merekomendasikan optimasi ambang sebagai alternatif terhadap penyeimbangan, yang
berada di luar cakupan penelitian ini.

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

Penelitian ini menggunakan tiga dataset deteksi fraud finansial dengan
karakteristik domain yang berbeda untuk menguji generalisasi metode lintas
domain. Dataset utama adalah Financial Fraud Detection Dataset (turunan simulator
PaySim), yaitu simulasi transaksi mobile money. Dataset kedua adalah
ULB Credit Card Fraud Detection Dataset, yaitu transaksi kartu kredit riil yang
fiturnya telah dianonimkan melalui Principal Component Analysis (PCA). Dataset
ketiga adalah Bank Account Fraud (BAF), yaitu dataset pembukaan rekening bank
terbitan Feedzai yang, berbeda dengan kedua dataset lainnya, menyediakan fitur
riil bernama dan bermakna semantik. PaySim berperan sebagai benchmark utama karena
skala dan karakteristik fiturnya, sementara ULB Credit Card dan BAF digunakan
sebagai dataset pembanding untuk menilai generalisasi model pada domain fraud
yang berbeda; BAF secara khusus menjadi dataset tempat analisis interpretabilitas
berbasis SHAP memiliki bobot makna paling kuat. Ketiga dataset diproses melalui
antarmuka pipeline yang identik sehingga seluruh model, skema agregasi, dan
metrik evaluasi dapat diterapkan tanpa modifikasi.

=== Karakteristik Dataset Utama (PaySim)

Dataset utama penelitian ini adalah Financial Fraud Detection Dataset yang
dipublikasikan pada platform Kaggle oleh Sriharsha Eedala. Dataset tersebut
merupakan turunan dari simulator PaySim @lopezrojas2016paysim, yaitu simulator
transaksi mobile money yang dikembangkan berdasarkan log transaksi nyata dari
sebuah perusahaan jasa keuangan di Afrika. Dataset ini dipilih karena memenuhi
tiga kriteria yang
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
  caption: [Karakteristik Dataset PaySim],
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
  caption: [Deskripsi Fitur Dataset PaySim],
) <tab-3-2>

=== Karakteristik Dataset Kedua (ULB Credit Card)

Dataset kedua penelitian ini adalah Credit Card Fraud Detection Dataset yang
dipublikasikan pada platform Kaggle oleh Machine Learning Group Université Libre
de Bruxelles (ULB). Dataset tersebut berisi transaksi kartu kredit riil nasabah
di Eropa selama dua hari pada September 2013. Berbeda dengan PaySim yang bersifat
sintetis, dataset ini merepresentasikan pola fraud kartu kredit yang nyata,
sehingga berfungsi sebagai pembanding untuk menilai generalisasi model pada
domain fraud yang berbeda. Karakteristik utama dataset disajikan pada @tab-3-3.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Kaggle (mlg-ulb/creditcardfraud, ULB)],
    [Jenis data], [Transaksi kartu kredit tabular (fitur ter-PCA)],
    [Jumlah baris], [284.807 transaksi],
    [Jumlah fitur], [31 kolom (30 fitur + 1 label)],
    [Label target], [Class (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 0,172% (492 transaksi fraud)],
    [Tipe fitur], [Numerik (Time, Amount, dan V1–V28 hasil PCA)],
  ),
  caption: [Karakteristik Dataset ULB Credit Card],
) <tab-3-3>

Struktur fitur dataset ini berbeda secara fundamental dari PaySim. Fitur V1
hingga V28 merupakan komponen hasil transformasi PCA yang telah dianonimkan untuk
menjaga kerahasiaan informasi nasabah, sehingga makna aslinya tidak dipublikasikan
dan nilainya sudah terstandardisasi. Hanya dua fitur yang tersaji dalam skala
asli, yaitu Time (selisih waktu dalam detik terhadap transaksi pertama) dan Amount
(nominal transaksi). Konsekuensinya, tahap preprocessing untuk dataset ini jauh
lebih ringkas dibandingkan PaySim: tidak diperlukan pembersihan kolom identifier,
one-hot encoding, maupun feature engineering saldo. Hanya fitur Time dan Amount
yang dinormalisasi menggunakan StandardScaler (di-fit pada training set saja),
sedangkan V1–V28 dibiarkan apa adanya karena telah terstandardisasi. Setelah
proses ini, dataset menghasilkan vektor berdimensi 30 fitur yang dikonsumsi oleh
seluruh model secara identik dengan pipeline PaySim.

=== Karakteristik Dataset Ketiga (Bank Account Fraud)

Dataset ketiga penelitian ini adalah Bank Account Fraud (BAF) yang dipublikasikan
oleh Feedzai pada NeurIPS 2022 @jesus2022baf dan tersedia di platform Kaggle.
BAF berisi data aplikasi pembukaan rekening bank daring, dengan label fraud yang
menandai aplikasi yang teridentifikasi sebagai penipuan identitas. Penelitian ini
menggunakan varian Base. Karakteristik utama dataset disajikan pada @tab-baf-char.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Kaggle (sgpjesus/bank-account-fraud-dataset-neurips-2022)],
    [Jenis data], [Aplikasi pembukaan rekening bank tabular],
    [Jumlah baris], [1.000.000 aplikasi],
    [Jumlah fitur], [32 kolom (31 fitur + 1 label)],
    [Label target], [fraud_bool (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 1,10% (11.029 aplikasi fraud)],
    [Tipe fitur], [Numerik dan kategorikal (fitur riil bernama)],
  ),
  caption: [Karakteristik Dataset Bank Account Fraud (BAF)],
) <tab-baf-char>

Berbeda dengan PaySim yang memiliki sedikit fitur dan ULB yang fiturnya
teranonimkan melalui PCA, BAF menyediakan fitur riil yang bernama dan bermakna
semantik, seperti income, customer_age, credit_risk_score, proposed_credit_limit,
serta sejumlah fitur velocity dan riwayat alamat. Kondisi ini menjadikan BAF
sebagai dataset yang paling relevan untuk analisis interpretabilitas berbasis
SHAP, karena kontribusi setiap fitur dapat ditafsirkan secara langsung dalam
konteks domain. Dataset ini memuat lima fitur kategorikal (payment_type,
employment_status, housing_status, source, dan device_os) dan selebihnya
merupakan fitur numerik. Rincian penanganan preprocessing BAF, yang mengikuti
alur bertipe PaySim (encoding kategorikal dan scaling), diuraikan pada Subbab
Perancangan Tahap Preprocessing.

=== Pembagian dan Penggunaan Dataset

Pembagian dataset dirancang dalam dua tingkat (two-level split) untuk
merefleksikan konteks penelitian yang melibatkan baseline terpusat sekaligus
simulasi federated learning. Tingkat pertama merupakan pembagian global yang
dilakukan satu kali pada keseluruhan dataset, sedangkan tingkat kedua merupakan
partisi training set ke seluruh client untuk skenario federated. Skema pembagian
dan partisi yang sama diterapkan secara identik pada ketiga dataset (PaySim,
ULB Credit Card, dan BAF), termasuk proporsi split, stratifikasi berdasarkan
label, dan penetapan random seed, sehingga hasil antar dataset dapat dibandingkan
secara setara. Uraian berikut menggunakan PaySim sebagai contoh.

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

Rangkaian langkah di atas berlaku untuk dataset utama PaySim. Untuk dataset ULB
Credit Card, tahap preprocessing jauh lebih ringkas sebagaimana diuraikan pada
Subbab Karakteristik Dataset Kedua, yaitu tanpa pembersihan identifier, one-hot
encoding, maupun feature engineering, dan hanya menormalisasi fitur Time dan
Amount karena V1–V28 telah terstandardisasi melalui PCA.

Dataset BAF mengikuti alur bertipe PaySim karena memuat fitur kategorikal dan
numerik pada skala mentah. Kolom device_fraud_count dihapus karena bernilai
konstan nol di seluruh dataset sehingga tidak membawa informasi. Kolom month
dikeluarkan dari himpunan fitur dan disimpan sebagai kolom pendamping (bukan
fitur), karena prevalensi fraud pada BAF berubah antar bulan sementara pembagian
data bersifat stratified-random dan bukan temporal; menyertakan month sebagai
fitur akan membuat model mempelajari prevalensi per-bulan dan menerapkannya pada
sampel uji dari bulan yang sama, sehingga menggelembungkan AUPRC secara artifisial.
Lima fitur kategorikal di-encode menggunakan one-hot encoding dengan daftar
kategori tetap agar himpunan kolom identik di seluruh client dan split. Lima
kolom numerik menggunakan nilai −1 sebagai sentinel "tidak tersedia"
(prev_address_months_count, bank_months_count, current_address_months_count,
session_length_in_minutes, dan device_distinct_emails_8w); untuk setiap kolom
tersebut ditambahkan indikator biner _missing_, nilai −1 diganti menjadi kosong,
kemudian diimputasi dengan median yang dihitung hanya pada training set. Kolom
yang memang bernilai negatif secara wajar (intended_balcon_amount, velocity_6h,
dan credit_risk_score) tidak diperlakukan sebagai sentinel. Perlu dicatat bahwa
prev_address_months_count memiliki tingkat ketidaktersediaan sekitar 71%,
sehingga setelah imputasi median kolom tersebut menjadi hampir konstan dan
sebagian besar sinyalnya justru terkandung pada indikator _missing_-nya; hal ini
merupakan properti data yang diketahui, bukan anomali. Setelah one-hot encoding
dan penambahan indikator, seluruh matriks fitur dinormalisasi menggunakan
StandardScaler yang di-fit pada training set saja, menghasilkan vektor berdimensi
55 fitur (24 numerik + 5 indikator _missing_ + 26 kolom one-hot) yang dikonsumsi
oleh seluruh model secara identik.

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
prinsip privasi FL, yaitu data tidak boleh meninggalkan client.

Aturan penyeimbangan ditetapkan secara *seragam* untuk ketiga dataset:
parameter sampling_strategy sebesar 0,01 (target proporsi minoritas terhadap
mayoritas 1:100), dengan k_neighbors = 5, diterapkan lokal per client, dan
dilewati bila sebuah client memiliki kurang dari 6 sampel minoritas atau telah
memenuhi target. Prevalensi dasar ketiga dataset berbeda hingga sekitar 8,5 kali
(PaySim 0,13%, ULB 0,172%, BAF 1,10%), sehingga tidak ada konfigurasi yang
"identik" dalam segala pengertian sekaligus — seseorang dapat menahan parameter
tetap konstan, atau endpoint tetap konstan, atau penguatan multiplikatif tetap
konstan, tetapi tidak ketiganya bersamaan. Penelitian ini menahan *parameter*
tetap konstan, sehingga diperoleh satu aturan tunggal tanpa penyetelan
per-dataset yang perlu dijustifikasi. Nilai 0,01 tidak memiliki optimum yang
diturunkan dari literatur — tidak ada optimum semacam itu — dan merupakan pilihan
pragmatis.

Penyeimbangan penuh (1:1) ditolak dengan alasan yang telah diberikan pada subbab
ini, yaitu sampel sintetis tidak boleh mendominasi partisi lokal. Sebagai
gambaran konkret, pada skenario IID sebuah client PaySim dengan sekitar 1.160
sampel fraud nyata dan sekitar 890.000 sampel mayoritas akan membutuhkan sekitar
889.000 sampel sintetis untuk mencapai 1:1, yakni multiplier sekitar 766 kali —
partisi lokalnya akan menjadi mayoritas data sintetis.

Konsekuensi dari aturan tunggal ini terungkap pada sensus per-client (disajikan
pada Subbab Implementasi Modul SMOTE Lokal). Karena prevalensi global BAF telah
melampaui target 1:100, aturan tersebut menjadi tidak beroperasi pada BAF dalam
skenario IID — seluruh client dilewati melalui kondisi `target_met`, sehingga
arm dengan-SMOTE menjadi identik dengan arm tanpa-SMOTE. Pada skenario Dirichlet
$alpha = 0.5$, ketimpangan partisi mendorong client yang starved jauh di bawah
target, sehingga SMOTE justru menyala secara selektif tepat pada partisi-partisi
sparse tersebut. Hal ini disajikan sebagai temuan: *sebuah target rasio absolut
yang tetap menjadi tidak beroperasi begitu prevalensi dasar suatu dataset
melampauinya, sehingga perlakuan efektif dalam setting federated ditentukan
secara bersama oleh target dan partisi, bukan oleh target semata.*

Ambang skip sebesar k_neighbors + 1 = 6 merupakan persyaratan algoritmik SMOTE
@chawla2002smote (interpolasi K-Nearest Neighbors membutuhkan minimal sekian
tetangga minoritas), bukan ambang berbasis bukti mengenai kapan SMOTE masih
valid; ambang berbasis bukti semacam itu tidak tersedia dalam literatur. Client
yang telah memenuhi atau melampaui target juga tidak dioversample karena SMOTE
hanya menambah sampel minoritas. Karena kedua ambang tersebut bersifat
per-client, penerapan SMOTE tidak selalu seragam antar client pada skenario
Non-IID — sebagian client dilewati sementara yang lain melakukan oversampling —
dan kondisi ini dicatat eksplisit agar studi ablasi tidak salah ditafsirkan
sebagai kondisi aktif/nonaktif yang seragam.

Dalam kerangka ini, konfigurasi *tanpa* koreksi diperlakukan sebagai baseline dan
SMOTE sebagai intervensi yang diuji, mengikuti temuan
#cite(<goorbergh2022harm>, form: "prose") bahwa koreksi imbalance belum tentu
diperlukan; penekanan ini bersifat penamaan kerangka saja dan tidak mengubah
rancangan eksperimen. Studi ablasi dengan dan tanpa SMOTE dilakukan untuk
mengisolasi pengaruh teknik ini.

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
model dilatih dengan empat skema agregasi sebagaimana diringkas pada @tab-3-4.

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
) <tab-3-4>

=== Perancangan Modul Evaluasi

Pengukuran performa dilakukan pada test set terpusat untuk seluruh skenario
eksperimen. Empat metrik utama dihitung: AUPRC (utama), F1-score, Precision, dan
Recall. Setiap eksperimen dijalankan sebanyak tiga kali dengan random seed yang
berbeda, dan hasil akhir dilaporkan sebagai rata-rata dengan deviasi standar.

Sebagai pelengkap keempat metrik utama tersebut, penelitian ini juga melaporkan
*Recall\@5%FPR*, yaitu proporsi transaksi fraud yang berhasil dideteksi sembari
membatasi false positive rate (FPR) pada paling banyak 5% — anggaran false alarm
yang ditetapkan sebagai parameter modul evaluasi (`TARGET_FPR = 0,05`) dan dapat
diubah bila kebijakan operasional menuntut anggaran berbeda. Metrik ini
diperoleh dengan membangun kurva ROC dari skor prediksi model, lalu memilih titik
operasi (operating point) dengan FPR $lt.eq 0,05$ yang menghasilkan Recall (TPR)
tertinggi — setara dengan ambang terendah yang masih memenuhi anggaran 5% false
alarm. Karena kerangka evaluasi tidak melakukan interpolasi pada metrik berbasis
ROC, titik yang dilaporkan merupakan titik operasi nyata yang dapat dijalankan
model, dan FPR aktual pada titik tersebut (umumnya sedikit di bawah 5%) turut
dilaporkan bersama ambangnya. Berbeda dari AUPRC yang meringkas kualitas
peringkat pada seluruh ambang, Recall\@5%FPR bersifat *operasional dan
spesifik-ambang*: metrik ini menjawab berapa banyak fraud yang tertangkap pada
anggaran false alarm tetap yang layak-terapkan, sehingga melengkapi — bukan
menggantikan — AUPRC. Metrik ini bersifat berbasis peringkat sehingga valid baik
untuk keluaran probabilitas maupun margin fungsi keputusan SVM. Nilainya
deterministik; pada kasus degeneratif (label satu kelas) metrik dikembalikan
sebagai `NA` alih-alih memicu pembagian dengan nol.

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
#cite(<lundberg2020treeshap>, form: "prose") dengan mode `interventional` dan
background lokal per-client diaplikasikan pada GBM dan XGBoost. Mode `interventional`
dipilih menggantikan `tree_path_dependent`: mode `tree_path_dependent` bersifat
bebas-background, sehingga untuk satu model global yang sama importance tiap client
menjadi identik dan stabilitas antar-client model pohon bernilai 1,0 secara trivial
— yang akan mengeluarkan model pohon dari analisis RQ3. Mode `interventional`
memanfaatkan background lokal tiap client sehingga menghasilkan variasi antar-client
yang nyata. Asumsi independensi fitur pada mode ini adalah asumsi yang sama yang
telah didokumentasikan untuk LinearSHAP dan KernelSHAP (lihat batasan di bawah),
sehingga tidak memperkenalkan kelas batasan baru. Varian LinearSHAP diaplikasikan
pada Logistic Regression
*dan* Support Vector Machine — keduanya model linear — karena memberikan exact
Shapley values berbentuk tertutup dengan biaya rendah; ini mengoreksi rancangan
awal yang memakai KernelSHAP untuk SVM. Khusus SVM, kuantitas yang dijelaskan
adalah margin fungsi keputusan (SVM hinge tidak mengeluarkan probabilitas), bukan
log-odds; ranking tetap komparabel antar-model namun magnitudonya tidak.

Varian KernelSHAP yang model-agnostik diaplikasikan pada FedXGBllr serta pada
kedua model deep, FFD dan BERT. FedXGBllr dijelaskan model-agnostik karena kepala
CNN-nya tidak linear terhadap keluaran pohon sehingga dekomposisi per-pohon eksak
tidak berlaku (lihat Subbab Implementasi Modul Evaluasi). FFD dan BERT dijelaskan
dengan KernelSHAP karena estimator berbasis gradien gagal memenuhi aksioma local
accuracy pada arsitektur tersebut — bukti empirisnya dibahas pada @sec-hasil-rq3.
Seluruh atribusi dihitung pada skala log-odds (margin untuk SVM), mengikuti
#cite(<sundararajan2020many>, form: "prose") yang menegaskan bahwa tidak ada
penjelasan Shapley tunggal bagi sebuah model — atribusi berbeda menurut apakah
yang dijelaskan adalah probabilitas, log-odds, atau keputusan; log-odds dipilih
seragam sebagai luaran mentah TreeSHAP, skala natural LR, dan skala tempat
argumen aditivitas FedXGBllr berlaku. Khusus FedXGBllr, log-odds diambil langsung
dari aktivasi pra-Sigmoid kepala CNN (keluaran lapisan linear sebelum Sigmoid),
bukan dari $"logit"(p)$ atas probabilitas keluaran. Pada dataset yang
probabilitas keluarannya terkompresi hingga $tilde.op 10^(-9)$ (PaySim),
$"logit"(p)$ terpotong oleh batas klip numerik sehingga seluruh prediksi jenuh
pada satu nilai konstan dan KernelSHAP menghasilkan atribusi nol; aktivasi
pra-Sigmoid merupakan log-odds eksak tanpa pemotongan dan konsisten dengan cara
FFD serta BERT dijelaskan (logit mentah pra-aktivasi).

Latar (background) KernelSHAP adalah 100 sampel data latih lokal pasca-SMOTE tiap
client yang diringkas menjadi 10 sentroid k-means; peringkasan ini menentukan
distribusi referensi dan wajib identik antara pengukuran floor dan produksi.
Jumlah evaluasi (nsamples) KernelSHAP ditetapkan bukan secara asumtif melainkan
melalui pengukuran agreement antar-seed: KernelSHAP dijalankan dua kali dengan
random seed berbeda pada satu client, lalu Spearman rank correlation antar kedua
vektor importance diukur pada nsamples ∈ {100, 500, 1000}. Nilai terkecil yang
mencapai Spearman > 0,95 pada ketiga model yang dijelaskan KernelSHAP adalah
nsamples = 500, yang karena itu digunakan. Pengukuran pemilihan tersebut
dilakukan di bawah perilaku bawaan `l1_reg` pustaka — lihat paragraf berikut —
sehingga angka floor historisnya tidak dibandingkan langsung dengan floor pada
protokol final.

Seluruh pemanggilan KernelSHAP menonaktifkan seleksi fitur bawaan pustaka
(`l1_reg=False`). Sejak shap 0.47.0, nilai bawaan `l1_reg="num_features(10)"`
menjalankan regresi LARS yang mempertahankan paling banyak 10 fitur per sampel
dan menetapkan atribusi seluruh fitur lain tepat 0,0 — pada BAF (55 fitur)
berarti sekurang-kurangnya 45 nol eksak per sampel yang dijelaskan. Perilaku
tersebut membuat statistik peringkat didominasi ties, dan karena himpunan
sepuluh fitur yang lolos merupakan fungsi diskontinu dari undian koalisi,
agreement antar-seed ikut tertekan sehingga floor yang terukur membesar secara
artifisial. Penonaktifannya diverifikasi oleh guard regresi pada runner: sel
dengan jumlah fitur di atas 10 yang seluruh baris atribusinya memuat paling
banyak 10 nilai taknol menghentikan eksekusi dengan pesan kesalahan eksplisit.

Ketidakpastian estimator tidak dibaca terhadap satu floor global yang diukur
pada satu dataset lalu disiarkan ke dataset lain, melainkan diuji per sel.
Untuk setiap sel dan setiap client, vektor importance global dihitung pada dua
seed koalisi khusus SHAP (seed pelatihan tidak disentuh) dengan explanation
data dan background identik, menghasilkan floor per-client
$rho(g_c^((s_1)), g_c^((s_2)))$ untuk setiap client $c$ — reliabilitas
estimator pada data client itu sendiri. Perbandingan antar-client memakai
common random numbers: kedua client pada setiap pasangan berbagi seed koalisi
yang sama, sehingga nilai antar-client $rho(g_i^((s)), g_j^((s)))$, $i < j$,
dievaluasi pada undian koalisi identik dan deviasi run-to-run-nya mengecil
tanpa menggeser taksiran titiknya. Di bawah hipotesis nol bahwa seluruh client
berbagi satu vektor importance sejati, ke-$2K$ vektor tersebut bersifat
exchangeable sehingga pemasangan (client, seed) bersifat arbitrer; distribusi
null karenanya dibentuk oleh seluruh perfect matching dari $2K$ vektor menjadi
$K$ pasangan — untuk $K = 5$ berjumlah $(2K - 1)!! = 945$ dan dienumerasi
lengkap, menghasilkan uji eksak dengan nilai p minimum 1/945. Statistik ujinya
adalah selisih rerata korelasi dalam-pasangan terhadap rerata korelasi
luar-pasangan; uji permutasi gabungan (pooling) tidak dipakai karena nilai
antar-client dihitung dari hanya lima vektor client sehingga saling bergantung
kuat. Rancangan seed-berulang ini mengikuti preseden pengujian stabilitas LIME
oleh #cite(<visani2022stability>, form: "prose") dan memperluasnya ke
perbandingan antar-client federatif. Koreksi multiplisitas Benjamini–Hochberg
diterapkan pada seluruh sel kernel multi-client; sebuah sel dinyatakan
berbeda-antar-client bila p terkoreksi $lt.eq 0,05$, dan rasio disatenuasi
between/floor dilaporkan sebagai indikator sekunder yang bersifat aproksimatif.
Ringkasan hasil memuat kolom floor per sel (`floor_mean`, `floor_min`),
statistik antar-client (`between_mean`, `between_sd`), selisih `delta`,
`p_value` beserta `p_adj`, dan verdict per sel — menggantikan penanda
`below_floor` tunggal, yang merangkum keliru dua kasus berlawanan: nilai
antar-client sedikit di bawah floor memang tak terbedakan dari noise estimator,
tetapi nilai yang berada jauh di bawah floor justru bukti melawan hipotesis
nol, yaitu sinyal ketidaksepakatan antar-client yang nyata. Skema pengukuran
tersebut beserta konsekuensinya diringkas pada @fig-3-two-seeds-gap.

#figure(
  image("resources/fig-3-two-seeds-gap.png", width: 82%),
  caption: [Skema pengukuran dua-seed. Panel A: setiap client menghasilkan dua vektor importance pada dua seed koalisi, sehingga kesepakatan dalam-client menjadi floor dan perbandingan antar-client dipasangkan pada seed yang sama (common random numbers). Panel B: sebaran kedua ukuran pada satu sel nyata (BAF BERT Dirichlet tanpa SMOTE). Di bawah hipotesis nol kedua sebaran berimpit, sehingga floor tidak dapat dipakai sebagai ambang deteksi dan digantikan oleh uji exchangeability eksak atas seluruh 945 perfect matching.],
) <fig-3-two-seeds-gap>

Pada setiap client, komputasi SHAP menghasilkan vektor feature importance lokal
yang diperoleh melalui rerata absolut SHAP values pada seluruh sampel explanation
data. Vektor ini merefleksikan seberapa besar pengaruh setiap fitur terhadap
prediksi model global menurut perspektif client yang bersangkutan. Vektor feature
importance dari seluruh client kemudian dihimpun di server simulasi untuk
dianalisis lebih lanjut menggunakan empat jenis komputasi statistik yang saling
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
mendekati 0 mengindikasikan interpretasi yang tidak stabil. Perlu ditegaskan
perbedaan antara korelasi nol dan korelasi tak-terdefinisi: bila vektor
importance sebuah client konstan atau seluruhnya nol — misalnya ketika KernelSHAP
jenuh pada regime probabilitas terkompresi — Spearman tidak terdefinisi dan
dikembalikan sebagai nilai kosong (nan), bukan dipaksa menjadi 0, karena nol
berarti "client sepenuhnya tidak sepakat" sedangkan tak-terdefinisi berarti
"peringkat memang degeneratif". Sel semacam itu ditandai sebagai *undefined*
beserta alasannya dan tidak menghasilkan metrik stabilitas apa pun; jika tidak,
Jaccard maupun Kuncheva pada peringkat yang seragam-akibat-ties akan keliru
menampilkan kesepakatan sempurna (1,0).

Komputasi ketiga adalah Jaccard similarity rerata pada lima fitur teratas (top-5)
antar seluruh pasangan client. Metrik ini mengukur proporsi fitur penting yang
sama-sama muncul pada peringkat lima teratas di dua client yang dibandingkan.
Pemilihan top-5 didasarkan pada pertimbangan praktis bahwa auditor dan regulator
umumnya hanya meninjau sejumlah kecil fitur teratas dalam proses validasi model,
sehingga kesepakatan antar client mengenai identitas fitur paling berpengaruh
menjadi indikator yang relevan secara operasional.

Namun Jaccard\@5 tidak terkoreksi terhadap peluang: nilai harapannya di bawah
seleksi acak menyusut seiring bertambahnya jumlah fitur, sehingga sebuah nilai
Jaccard top-5 tidak sebanding antar dataset berdimensi berbeda
@nogueira2018stability. Karena ketiga dataset memiliki jumlah fitur berbeda
(≈13, 30, dan 55), penelitian ini menambahkan indeks konsistensi Kuncheva
@kuncheva2007stability sebagai metrik keempat — ukuran terkoreksi-peluang yang
bernilai ≈0 pada seleksi acak berapa pun dimensinya dan 1 pada kesepakatan
sempurna. Setiap klaim yang membandingkan stabilitas antar-dataset bersandar pada
indeks Kuncheva, sedangkan Jaccard\@5 dipertahankan untuk kesinambungan dengan
argumen auditor di atas; Spearman tidak terpengaruh karena null-nya 0 tanpa
bergantung dimensi.

Tiga perluasan melengkapi keempat metrik tersebut. Pertama, stabilitas tidak
diringkas pada satu nilai k saja: indeks Kuncheva dihitung pada seluruh
k = 1…M dan dibandingkan terhadap pita floor per-client pada k yang sama,
sehingga kedalaman peringkat yang masih terbaca — titik ketika kurva
antar-client meninggalkan pita floor-nya — teridentifikasi eksplisit; profil
ini terbanding antar dataset berdimensi berbeda berkat koreksi peluangnya
@nogueira2018stability. Korelasi peringkat berbobot magnitudo (bobot rerata
|SHAP| per fitur) turut dilaporkan agar ekor fitur beratribusi hampir nol
tidak mendominasi statistik peringkat penuh. Kedua, untuk memisahkan dua
sumber divergensi yang terkonfundasi — model global berperilaku berbeda pada
wilayah data yang ditempati sebuah client, versus distribusi background lokal
client itu sendiri yang berbeda — sel-sel terpilih dijalankan pada dua arm yang
masing-masing hanya membiarkan satu faktor bervariasi antar client. Arm baku
menahan explanation data tetap (subset test terpusat yang identik bagi seluruh
client) dan membiarkan background bervariasi per client, sehingga divergensi
yang terukur berasal dari perbedaan distribusi background. Arm kedua melakukan
kebalikannya: background disatukan menjadi satu background bersama hasil
penggabungan seluruh background lokal — mengikuti motivasi background federatif
#cite(<ducange2026fedshap>, form: "prose") — sementara setiap client menjelaskan
sampel dari partisi lokalnya sendiri, sehingga divergensi yang terukur berasal
dari perbedaan wilayah data antar client. Menyatukan background sekaligus
menyeragamkan explanation data akan membuat seluruh client menerima masukan yang
identik sehingga sumbu client runtuh dan setiap ukuran kesepakatan bernilai 1,0
secara struktural; kondisi tersebut dicegat oleh guard yang menandai sel
demikian sebagai tak-terdefinisi tanpa metrik stabilitas apa pun. Ketiga, khusus PaySim KernelSHAP
dibuat eksak: dengan $M = 13$ fitur, nilai nsamples sebesar
$2^(13) - 2 = 8190$ mengenumerasi seluruh bobot kernel tanpa satu pun undian
acak, dan bila kelima kolom one-hot `type` dikelompokkan sebagai satu pemain
Shapley ($M$ efektif sama dengan 9) enumerasi lengkap hanya membutuhkan 510
evaluasi per sampel — setara anggaran produksi. Kedua rute eksak dijalankan
dan saling memvalidasi; pengelompokan yang sama pada BAF menghasilkan $M$
efektif 29 sehingga enumerasi lengkap tetap di luar jangkauan dan manfaat
pengelompokan di sana terbatas pada reduksi variansi. Keeksakan kedua rute
tersebut tidak diandaikan melainkan diukur pada setiap sel: selisih maksimum
absolut antara matriks atribusi kedua seed koalisi dicatat sebagai besaran
tersendiri, wajib bernilai nol eksak pada sel yang mengenumerasi seluruh
koalisi, dan pada sel tersampel besaran yang sama berperan sebagai ukuran galat
estimator dalam satuan atribusi.

Sebagai batasan, LinearSHAP, KernelSHAP, dan TreeSHAP mode `interventional`
sama-sama mengasumsikan independensi fitur sehingga koalisi yang disampel dapat
membentuk kombinasi mustahil — khususnya blok one-hot yang secara struktural hanya
bernilai 1 pada tepat satu kolom (26 dari 55 kolom pada BAF, 5 pada PaySim; ULB
bersih karena seluruhnya komponen PCA). Hal ini dapat menggeser atribusi ke luar
domain pelatihan @aas2021explaining dan didokumentasikan sebagai batasan;
pengelompokan one-hot pada tingkat eksak PaySim di atas menanganinya sebagian —
blok `type` diperlakukan sebagai satu pemain sehingga koalisi mustahil di dalam
blok tidak lagi tersampel — sedangkan pada dataset lain batasan ini tetap
berlaku. Karena model pohon kini memakai mode
`interventional` (bukan `tree_path_dependent`), asumsi ini berlaku seragam pada
seluruh explainer yang bergantung-background dan tidak menambah kelas batasan baru.

Kombinasi keempat metrik ini memberikan gambaran komplementer mengenai
karakteristik explainability model. Rerata feature importance menunjukkan apa
yang diinterpretasikan sebagai penting, sedangkan Spearman, Jaccard, dan indeks
Kuncheva menunjukkan seberapa stabil interpretasi tersebut antar client
di bawah heterogenitas data. Keempat metrik dilaporkan untuk setiap kombinasi
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
komputasi yang spesifikasinya disajikan pada @tab-3-5.

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
    [Library XAI], [SHAP (shap) 0.49.1],
    [Library numerik & DataFrame], [NumPy, Pandas, SciPy],
    [Library visualisasi], [Matplotlib, Seaborn],
    [Pelacakan eksperimen], [Weights & Biases (wandb 0.15.12)],
    [Pengujian regresi], [pytest 9.1.1],
    [Version control], [Git + GitHub],
  ),
  caption: [Spesifikasi Lingkungan Pengembangan],
) <tab-3-5>

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
penyeimbangan ditetapkan melalui parameter sampling_strategy secara seragam
sebesar 0,01 untuk ketiga dataset (lihat Subbab Perancangan Skema Class Imbalance
Handling); client yang telah mencapai atau melampaui target tersebut dilewati
karena SMOTE hanya menambah sampel kelas minoritas. Untuk mendukung analisis pada
Bab berikutnya, modul mencatat secara eksplisit pada setiap client (dan pada
setiap putaran untuk FFD yang menerapkan SMOTE ulang tiap putaran) jumlah sampel
sintetis relatif terhadap jumlah sampel minoritas riil, yaitu multiplier sintesis,
beserta status diterapkan atau dilewatinya SMOTE. Pencatatan ini memberikan bukti
empiris langsung mengenai potensi sintesis yang tidak representatif ketika sebuah
client hanya memiliki sedikit sampel fraud riil namun target memaksa pembangkitan
banyak sampel sintetis. Sebagai gambaran besaran risiko, pada partisi Non-IID kuat
($alpha = 0.5$) sebuah client dapat memperoleh hanya beberapa puluh sampel fraud
riil sementara target 1:100 tetap menuntut pembangkitan ribuan sampel sintetis,
sehingga multiplier sintesis dapat mencapai orde ratusan kali (terukur hingga
sekitar 185 kali pada kasus terburuk, lihat di bawah); kondisi ini merupakan
properti bawaan dari kombinasi target dan partisi yang diketahui sejak
perancangan, bukan anomali, dan justru menjadi alasan pencatatan multiplier
per-client dilakukan. Sebagai bagian integral dari studi ablasi, modul SMOTE dapat
dinonaktifkan secara global melalui parameter konfigurasi yang relevan, sehingga
memungkinkan pengamatan kontribusi murni teknik penanganan class imbalance
terhadap performa akhir model.

==== Diagnostik Geometri Sintesis SMOTE pada Client Terburuk BAF

Untuk memeriksa secara langsung apakah sintesis pada regime sampel minoritas kecil
menghasilkan penambahan informasi yang berarti, dilakukan diagnostik geometri pada
client dengan sampel fraud paling sedikit, yaitu client BAF dengan seed 42,
partisi Dirichlet $alpha = 0.5$, yang hanya memuat 21 sampel fraud riil di antara
391.355 baris. SMOTE dijalankan persis seperti pada pelatihan
(`sampling_strategy = 0,01`, `k_neighbors = 5`), lalu titik sintetis dibandingkan
dengan seed nyata pada proyeksi PCA dan t-SNE (@fig-3-4-smote-geometry).

#figure(
  image("resources/fig-3-4-smote-geometry-baf.png", width: 100%),
  caption: [Geometri sintesis SMOTE pada client terburuk BAF (seed 42, Dirichlet
  $alpha = 0.5$, client indeks 1, sampling_strategy = 0,01, random state SMOTE =
  43). Partisi penuh berisi 391.334 mayoritas nyata, 21 minoritas nyata, dan
  3.892 sintetis; yang diplot adalah 8.000 mayoritas dan 3.892 sintetis
  (subsampel untuk t-SNE) beserta seluruh 21 minoritas. Panel kiri t-SNE, panel
  kanan PCA. Titik minoritas nyata digambar paling akhir dan lebih besar agar
  tetap terlihat terhadap massa sintetis.],
) <fig-3-4-smote-geometry>

Untuk memisahkan pengaruh besaran target dari pengaruh jumlah seed, diagnostik
yang sama dijalankan pada dua nilai rasio pada client yang identik; hasilnya
disajikan pada @tab-smote-geometry.

#figure(
  kind: table,
  table(
    columns: (1.6fr, 1fr, 1fr),
    align: (left, right, right),
    table.header([*Kuantitas*], [*0,01 (konfigurasi eksperimen)*], [*0,10 (pembanding)*]),
    [Minoritas nyata (seed)], [21], [21],
    [Mayoritas nyata], [391.334], [391.334],
    [Sintetis minoritas], [3.892], [39.112],
    [Multiplier sintesis], [×185], [×1862],
    [Segmen terisi], [82 dari 210], [82 dari 210],
    [Titik per segmen (rerata)], [47,5], [477],
    [Residual on-segment (median)], [2,6e−7], [2,6e−7],
    [Tetangga nyata terdekat = mayoritas], [16,03%], [15,04%],
  ),
  caption: [Geometri sintesis SMOTE pada client terburuk BAF untuk dua nilai
  sampling_strategy. Jumlah seed, segmen, dimensionalitas, dan fraksi kontaminasi
  praktis tidak berubah; hanya volume sintetis yang berubah.],
) <tab-smote-geometry>

Client dengan 21 sampel fraud nyata lolos dari ambang 6, namun setiap titik
sintetis terletak pada segmen antara sebuah seed dan salah satu dari lima tetangga
minoritas terdekatnya — paling banyak sekitar 105 segmen berdimensi satu pada
ruang berdimensi 55, dan pada praktiknya hanya 82 segmen yang menampung seluruh
massa sintetis. Kelas minoritas pada partisi tersebut berbentuk *wireframe*, bukan
awan (cloud); residual on-segment yang mendekati nol (2,6e−7) mengonfirmasi bahwa
titik-titik benar-benar berada tepat pada segmen tersebut. Fraksi 16,03% titik
sintetis yang tetangga *nyata* terdekatnya adalah sampel mayoritas mewujudkan
fenomena pembangkitan di wilayah mayoritas yang dijelaskan @elreedy2024smote,
sedangkan salinan kolinear di sepanjang segmen mewujudkan penurunan variabilitas
dan korelasi terinduksi yang dilaporkan @blagus2013smote. Dalam terminologi
@weiss2004rarity, client ini mengalami absolute rarity, yang tidak dapat
disembuhkan oleh interpolasi.

Perbandingan dua rasio pada tabel mengubah klaim kausal menjadi pengukuran:
*volume* sintetis berskala dengan sampling_strategy (3.892 pada 0,01 versus 39.112
pada 0,10; 47,5 versus 477 titik per segmen), sedangkan jumlah segmen (82),
dimensionalitas, dan fraksi kontaminasi (~15–16%) tidak berubah. Kolapsnya
sintesis menjadi struktur satu dimensi ditentukan oleh jumlah seed, bukan oleh
rasio target. Karena di bawah partisi Dirichlet jumlah seed adalah fungsi dari
$alpha$, maka partisi Dirichlet-lah yang memproduksi regime sampel-kecil tempat
SMOTE sudah diketahui gagal — dalam FL cross-silo, jumlah minoritas bukan properti
tetap dataset melainkan produk partisi, sehingga $alpha$ secara langsung mengatur
seberapa jauh client sparse jatuh ke dalam regime tersebut.

Sensus per-client atas seluruh kombinasi dataset × skema × seed disajikan pada
@tab-minority-census, yang mengukur seberapa terekspos tiap dataset terhadap
regime ini.

#figure(
  kind: table,
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    table.header(
      [*Dataset*], [*Skema*], [*client < 10*], [*min*], [*median*],
      [*multiplier terburuk*], [*skip: target_met*],
    ),
    [PaySim], [IID], [0], [1069], [1155], [×7], [0/15],
    [PaySim], [Dirichlet α=0,5], [2], [2], [1080], [×30], [2/15],
    [PaySim], [Dirichlet α=1,0], [0], [88], [1030], [×123], [2/15],
    [PaySim], [Dirichlet α=5,0], [0], [249], [1164], [×41], [0/15],
    [ULB], [IID], [0], [53], [67], [×7], [0/15],
    [ULB], [Dirichlet α=0,5], [2], [4], [53], [×30], [1/15],
    [ULB], [Dirichlet α=1,0], [0], [17], [49], [×23], [2/15],
    [ULB], [Dirichlet α=5,0], [0], [41], [69], [×11], [0/15],
    [BAF], [IID], [0], [1494], [1540], [×0], [15/15],
    [BAF], [Dirichlet α=0,5], [1], [3], [1095], [×185], [8/15],
    [BAF], [Dirichlet α=1,0], [0], [128], [1016], [×13], [8/15],
    [BAF], [Dirichlet α=5,0], [0], [782], [1445], [×1], [8/15],
  ),
  caption: [Sensus minoritas per-client (sampling_strategy = 0,01), diagregasi
  atas 3 seed × 5 client = 15 instansi per baris. "client < 10" menghitung instansi
  dengan minoritas di bawah 10; "skip: target_met" menghitung instansi yang
  dilewati karena telah memenuhi target.],
) <tab-minority-census>

Sensus menegaskan dua hal. Pertama, dataset yang paling terekspos bukanlah yang
diduga semula: tidak ada satu pun sel yang menghasilkan client bernol minoritas,
namun ULB paling terekspos dalam pengertian berbeda — ia miskin minoritas secara
kronis di *seluruh* client (median hanya 53 lawan sekitar 1.080 pada PaySim dan
BAF), yaitu sekitar 1,8 kejadian per parameter (events per parameter), bukan hanya
pada ekornya. Rasio kejadian per parameter serendah ini berada jauh di bawah
ambang yang lazim dibahas pada literatur ukuran sampel model prediksi
(@vansmeden2016epv, @vansmeden2019samplesize, @riley2019minimum); nilai ini
dilaporkan sebagai deskriptor eksposur, bukan sebagai gerbang. Kedua,
sebuah target absolut yang tetap menjadi tidak beroperasi ketika prevalensi dasar
telah melampauinya: pada BAF IID seluruh 15 instansi dilewati melalui
`target_met`, sehingga arm dengan-SMOTE identik dengan arm tanpa-SMOTE, sementara
pada PaySim dan ULB kondisi ini nyaris tidak muncul.

=== Implementasi Pelatihan Model dan Skema Agregasi

Realisasi pelatihan keenam model dengan empat skema agregasi yang dirangkum pada
@tab-3-4 dibangun di atas kerangka kerja Flower, dengan setiap skema agregasi
diimplementasikan sebagai strategi terkustomisasi yang mewarisi antarmuka
strategi standar dari Flower. Pendekatan modular ini memungkinkan pertukaran
skema agregasi tanpa memodifikasi komponen lain pada pipeline sistem.

Terkait reproduktibilitas, dengan penetapan random seed yang konsisten, model
parametrik (LR, SVM) dan model berbasis pohon (GBM, FedXGBllr) bersifat
reproducible secara bit-per-bit, sedangkan model deep learning (FFD dan BERT)
hanya reproducible secara distribusi — nilai metriknya stabil antar-eksekusi
namun tidak identik bit-per-bit — karena operasi floating-point yang
non-asosiatif pada perangkat pelatihan; mode deterministik penuh pada model deep
learning sengaja tidak dipaksakan agar tidak mengorbankan kecepatan pelatihan.

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

*Seleksi Iterasi Berbasis Validation Set untuk GBM.* Setiap pelatihan GBM—baik
baseline terpusat maupun setiap model client pada skema federated—menjalani
seleksi iterasi (boosting prefix) pada validation set terpusat: model dilatih
penuh hingga max_iter = 100, kemudian prefix boosting dengan AUPRC validation
tertinggi dipertahankan ($k^*$ iterasi, $k^* lt.eq 100$). Mekanisme ini
menggunakan sinyal yang sama dengan best-model selection—AUPRC pada validation
set terpusat—namun diterapkan lintas-iterasi alih-alih lintas-client, sehingga
bersifat adaptif terhadap masing-masing arm SMOTE. Selubung ekivalensi aditif
boosting menjamin bahwa mempertahankan prefix $k^*$ identik dengan melatih ulang
model pada max_iter = $k^*$, sehingga tidak ada pelatihan tambahan yang
diperlukan. Jumlah iterasi terpilih dicatat pada kolom n_iter_selected setiap
baris hasil.

Justifikasi pemilihan mekanisme ini bersandar pada perbandingan dengan XGBoost.
Pada arm tanpa SMOTE dengan imbalance ekstrem, budget 100 iterasi penuh membuat
GBM overfit menjadi probabilitas jenuh (saturated) yang meruntuhkan ranking: pada
ULB test AUPRC GBM runtuh ke 0,18, sedangkan XGBoost pada data yang identik—dengan
50 pohon, subsample 0,8, dan eval_metric AUPRC—mencapai 0,84; pola yang sama namun
lebih ekstrem teramati pada PaySim. Selisih ini merupakan artefak budget boosting,
bukan temuan imbalance, sebagaimana ditegaskan oleh BAF: di sana GBM (0,161) telah
setara dengan XGBoost (0,157) tanpa saturasi, dan seleksi iterasi tidak mengubah
hasilnya. Seleksi berbasis validation set terpusat dipilih alih-alih opsi
`early_stopping='auto'` bawaan pustaka, karena opsi tersebut memotong 10%
validation holdout internal—sebuah split kedua tersembunyi di dalam salah satu
dari enam model yang merusak jaminan komparabilitas lapisan cache/hash bersama—dan
menurunkan performa arm SMOTE demi memperbaiki arm tanpa SMOTE. Sebaliknya, seleksi
prefix pada validation set terpusat memakai sinyal yang sudah digunakan skema
best-model selection dan tidak menambah split baru.

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

*Anggaran putaran yang asimetris untuk FedXGBllr.* FedXGBllr menggunakan 50 global
rounds mengikuti konfigurasi baseline hfedxgboost Flower, berbeda dari 20 rounds
untuk kelima model lainnya. Karena tahap CNN menerapkan early stopping pada
validation set terpusat, anggaran ini berperan sebagai batas atas dan bukan biaya
tetap: model global akhir adalah model dengan AUPRC validasi terbaik, bukan model
pada putaran ke-50. Membatasi setiap run FedXGBllr pada nilai terbaiknya dalam 20
putaran pertama mengubah AUPRC validasi paling banyak sebesar 2,2% dan sebesar
0,0% pada lima dari delapan run, sehingga asimetri anggaran putaran tidak
memengaruhi komparabilitas antar paradigma secara material. Agar interpretasi
tetap konsisten, setiap baris hasil mencatat `rounds_configured` (anggaran, yaitu
20 atau 50) berdampingan dengan `rounds_completed`, yakni jumlah putaran fit
federated yang benar-benar dieksekusi (di luar evaluasi awal pada putaran ke-0),
sehingga kolom tersebut bermakna sama untuk seluruh model.

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
@tab-3-6. Pemilihan nilai hyperparameter dilakukan secara terbatas, baik melalui
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
      [Global rounds (R)], [20 (50 untuk FedXGBllr, mengikuti baseline hfedxgboost Flower)],
      [Dirichlet $alpha$], [{0,5 ; 1,0 ; 5,0}],
      [Random seed], [42],
      [SMOTE: k_neighbors], [5],
      [SMOTE: sampling_strategy], [Per-dataset: 0,01 (1:100) untuk PaySim dan ULB; 0,10 (1:10) untuk BAF],

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
      [Seleksi iterasi], [Validation-set (prefix pemaksimum AUPRC, $k^* lt.eq 100$)],

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
) <tab-3-6>

=== Implementasi Modul Evaluasi dan SHAP

Realisasi modul evaluasi yang dirancang pada Subbab Perancangan Modul Evaluasi
mencakup dua komponen yang saling melengkapi, yaitu pengukuran performa model dan
analisis explainability, yang keduanya dieksekusi pada model global akhir setiap
skenario eksperimen.

Komponen pengukuran performa direalisasikan sebagai sekumpulan fungsi metrik
bersama pada modul `evaluation/metrics.py` yang dipakai identik oleh seluruh model
FL maupun baseline terpusat, sehingga skema penilaian seragam antar arm. AUPRC
dihitung bebas-ambang, sedangkan F1-score, Precision, dan Recall dihitung pada
ambang hasil penyetelan (max-F1 pada validation set). Metrik pelengkap
Recall\@5%FPR diimplementasikan pada fungsi `recall_at_fpr` yang membangun kurva
ROC melalui `sklearn.metrics.roc_curve`, memilih titik operasi dengan FPR
$lt.eq 0,05$ yang memberikan Recall tertinggi tanpa interpolasi, lalu melaporkan
Recall, ambang, dan FPR aktual pada titik tersebut. Fungsi ini bersifat
deterministik dan aman terhadap kasus degeneratif (label satu kelas dikembalikan
sebagai `NA`, tanpa pembagian dengan nol). Nilai Recall\@5%FPR disurfacekan ke
seluruh jalur pelaporan — keluaran konsol per-ronde maupun final, log W&B, CSV
per-run (`test_recall_at_fpr`, `test_threshold_at_fpr`, `test_actual_fpr`, serta
`best_val_recall_at_fpr` dan kolom per-ronde `val_recall_at_fpr`), dan tabel
ringkasan agregat — berdampingan dengan AUPRC tanpa mengubah komputasi AUPRC.

Metrik Recall\@5%FPR ditambahkan setelah sweep utama selesai, sehingga nilainya
di-*backfill* dengan memuat ulang model global final yang telah dibekukan pada
`results/models/` dan menghitung ulang skornya memakai loader baca-saja yang sama
dengan probe SHAP; tidak ada model yang dilatih ulang. Penambahan ini bersifat
post hoc dan dinyatakan demikian secara transparan, bukan bagian dari rancangan
awal. Kebenaran tiap baris dijaga oleh satu pengecekan: AUPRC yang dihitung ulang
wajib cocok dengan `test_auprc` tersimpan dalam toleransi $2 times 10^(-3)$, jika
tidak baris dilewati. Seluruh 96 sel lolos, sehingga setiap nilai yang di-backfill
terbukti berasal dari keadaan model yang menghasilkan baris tersebut, dan seluruh
sel lain diverifikasi identik secara bita setelah penulisan ulang (kolom hash tidak
berubah).

Dua keterbatasan rekonstruksi dicatat. Pertama, `best_val_recall_at_fpr` hanya
tersedia untuk run terpusat: run federated memilih model final pada putaran
terbaiknya sedangkan model per-putaran tidak dipersistensi, sehingga nilai sisi
validasi tidak dapat direkonstruksi dan sel tersebut dibiarkan kosong untuk run FL.
Kedua, `val_recall_at_fpr` per-putaran tidak dapat di-backfill sama sekali karena
tidak ada artefak per-putaran yang disimpan. Kedua kolom akan terisi secara wajar
pada eksekusi berikutnya karena metrik ini kini menjadi bagian tetap dari pipeline
evaluasi.

Komponen analisis explainability merealisasikan kerangka pengukuran yang telah
dirancang pada Subbab Perancangan Modul Evaluasi dengan mengacu pada konfigurasi
data dan varian explainer yang telah ditetapkan. Implementasi dilakukan
menggunakan pustaka SHAP versi 0.49.1 — dipin eksak pada `requirements.txt`
karena perilaku bawaan `l1_reg` KernelExplainer berubah pada versi 0.47.0 —
serta diintegrasikan dengan pipeline evaluasi sehingga komputasi
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

Implementasi explainer mengikuti pemetaan pada Subbab Perancangan Modul Evaluasi,
dengan satu koreksi terhadap rancangan awal FedXGBllr. TreeSHAP diaplikasikan pada
GBM dan XGBoost melalui TreeExplainer dengan mode `interventional` dan background
lokal per-client, pada model hasil seleksi iterasi (prefix $k^*$ pohon) — yakni
persis model yang dijelaskan oleh metriknya. Mode `interventional` dipakai
menggantikan `tree_path_dependent` agar background lokal tiap client menghasilkan
variasi antar-client yang nyata; mode bebas-background akan membuat stabilitas
antar-client model pohon bernilai 1,0 secara trivial.

Rancangan awal menjelaskan FedXGBllr melalui dekomposisi TreeSHAP per-pohon yang
dibobot oleh learnable learning rates, dengan asumsi $phi_j (f) = sum_t w_t dot
phi_j (h_t)$. Dekomposisi tersebut hanya eksak menurut aksioma linearitas Shapley
apabila kepala agregator linear terhadap keluaran pohon. Arsitektur kepala CNN
FedXGBllr adalah `conv1d → flatten → ReLU → Linear → Sigmoid`; keberadaan ReLU
(dan Sigmoid pada tahap akhir) membuat luaran tidak linear terhadap keluaran
pohon, sehingga syarat linearitas tidak terpenuhi dan dekomposisi per-pohon tidak
berlaku. FedXGBllr karenanya dijelaskan secara model-agnostik dengan KernelSHAP
atas fitur asli, dengan memperlakukan komposisi tree-ensemble dan CNN sebagai satu
fungsi tunggal; koreksi ini tidak bergantung pada hasil SHAP dan berlaku semata
karena arsitektur model.

LinearSHAP diaplikasikan pada Logistic Regression dan pada SVM linear melalui
LinearExplainer, sesuai pemetaan pada Subbab Perancangan Modul Evaluasi (untuk
SVM kuantitas yang dijelaskan adalah margin fungsi keputusan). Khusus untuk
KernelSHAP, jumlah evaluasi fungsi ditetapkan nsamples = 500 dari pengukuran
agreement antar-seed — bukan nilai bawaan pustaka — dan seleksi fitur bawaan
dinonaktifkan (`l1_reg=False`) sebagaimana dirancang; keduanya diberlakukan
seragam pada FedXGBllr, FFD, dan BERT.

Komputasi feature importance dijalankan secara independen pada setiap client
setelah model global akhir tersedia. Setiap client menghitung SHAP values untuk
seluruh sampel explanation data pada dua seed koalisi khusus SHAP — explanation
data dan background identik antar seed, dengan common random numbers antar
client — kemudian merangkumnya menjadi vektor feature importance lokal melalui
rerata absolut, menghasilkan dua vektor per client yang sekaligus menjadi dasar
floor per-client. Vektor-vektor ini kemudian dikirimkan ke server simulasi
untuk dihimpun menjadi matriks feature importance antar client, yang menjadi
dasar seluruh analisis komparatif berikutnya.

Pada server simulasi, komputasi statistik sesuai rancangan pada Subbab
Perancangan Modul Evaluasi direalisasikan pada dua modul murni numerik. Modul
`evaluation/shap_stability.py` menghitung rerata konsensus, Spearman rank
correlation antar-client, Jaccard\@5, dan indeks Kuncheva; modul
`evaluation/shap_inference.py` menghitung floor per-client dua-seed, uji
exchangeability eksak atas seluruh perfect matching (945 pada K = 5), koreksi
Benjamini–Hochberg, profil Kuncheva atas k, dan korelasi peringkat berbobot
magnitudo — keduanya bebas dependensi model sehingga teruji unit di luar
lingkungan GPU. Orkestrasi produksinya adalah `experiments/shap_rq3.py`, yang
menulis artefak per sel (matriks importance per client dan seed, berkas
`stability.json` berisi seluruh statistik beserta provenance, dan profil
stabilitas per k) serta ringkasan `shap_summary_v2.csv` dengan kolom floor,
between, delta, nilai p mentah dan terkoreksi, selisih maksimum antar-seed, dan
verdict per sel. Empat guard melindungi pelaporan dari nilai kesepakatan yang
bersifat artefak: guard degenerasi mencatat sel yang vektor atribusinya runtuh
menjadi nol atau konstan sebagai undefined tanpa metrik stabilitas; guard
keruntuhan sumbu client menandai sel yang seluruh vektor client-nya identik pada
satu seed — kondisi yang membuat setiap ukuran kesepakatan bernilai 1,0 secara
struktural — juga sebagai undefined; guard regresi `l1_reg` menghentikan
eksekusi bila pola seleksi-sepuluh-fitur masih terdeteksi pada keluaran; dan
guard keeksakan menghentikan sel mana pun yang dinyatakan bebas galat estimator
— baik karena memakai explainer eksak maupun karena nsamples-nya mengenumerasi
seluruh koalisi — namun selisih antar-seednya tidak persis nol. Guard terakhir
menutup satu celah pelaporan: floor bernilai 1,0 pada tingkat eksak kini
merupakan hasil pengukuran per sel, bukan konsekuensi asumsi, dan status
enumerasi ditentukan oleh nsamples semata sehingga pengelompokan one-hot tidak
lagi cukup untuk menyatakan sebuah sel eksak. Keluaran tahap sebelum
perbaikan `l1_reg` dipertahankan utuh pada direktori terpisah sehingga setiap
angka yang berubah dapat dilaporkan berdampingan dengan nilai lamanya.

Hasil pengukuran dilaporkan untuk setiap kombinasi model, skenario partisi, dan
penerapan SMOTE, kemudian disajikan dalam bentuk tabel komparatif yang
memungkinkan analisis lintas paradigma agregasi. Visualisasi pelengkap berupa
heatmap matriks feature importance antar client, profil stabilitas terhadap k
dengan pita floor per-client, serta summary plot SHAP digunakan untuk
mendukung interpretasi kualitatif. Stabilitas yang menurun seiring penurunan
parameter Dirichlet $alpha$ — kini diuji secara formal per sel, bukan dibaca
terhadap satu ambang tunggal — akan diinterpretasikan sebagai indikasi
sensitivitas model terhadap heterogenitas distribusi data antar client, yang
menjadi salah satu kontribusi orisinal penelitian ini terhadap diskursus
Explainable Federated Learning.

// ---------------------------------------------------------------------------
// BAB 4 — HASIL DAN PEMBAHASAN  (stub)
// ---------------------------------------------------------------------------

= HASIL DAN PEMBAHASAN

Keseluruhan eksperimen terdiri atas 108 sel yang terbentuk dari tiga dataset, enam
model, tiga kondisi partisi, dan dua arm SMOTE. Dari jumlah keseluruhan tersebut, 
12 sel dilewati karena tidak memenuhi syarat oversampling SMOTE untuk eksperimen 
ini sehingga tersisa 96 sel yang tereksekusi. Sel-sel yang dilewati itu terjadi karena tingkat fraud
BAF sebesar 1,10 persen telah melampaui target rasio eksperimen 1:100 sehingga pada kondisi
centralized dan IID setiap client sudah memenuhi target dan kedua arm menjadi
identik. Oleh karena itu, arm SMOTE untuk BAF hanya bermakna pada kondisi Dirichlet, 
tempat sebagian client kekurangan kasus fraud.

Seluruh eksperimen dijalankan pada satu seed, yaitu seed 42. Konsekuensinya, seluruh angka pada bab ini
merupakan estimasi titik tanpa ukuran variansi dan secara khusus variansi akibat
pengacakan partisi Dirichlet tidak terkuantifikasi sehingga selisih kecil antar
sel tidak dapat diklaim signifikan. Kedua, karena AUPRC memiliki batas bawah yang
setara dengan prevalensi kelas positif, perbandingan lintas dataset atas angka
AUPRC mentah bersifat menyesatkan. Baseline acak berbeda tajam antar dataset,
yaitu 0,00129 pada PaySim, 0,00173 pada ULB, dan 0,01103 pada BAF
#cite(<saito2015>). Seluruh pembahasan pada bab ini karena itu merujuk pada
baseline masing-masing dataset.

== Perbandingan Performa Antar Paradigma Agregasi

=== Performa AUPRC dan Recall\@5%FPR lintas dataset dan kondisi

Perbandingan performa antar paradigma agregasi diawali dari metrik diskriminasi
utama, yaitu AUPRC yang dihitung pada test set terpusat. @tab-4-auprc-ulb,
@tab-4-auprc-baf, dan @tab-4-auprc-paysim menyajikan nilai tersebut bagi setiap
model pada masing-masing dataset, dipilah menurut kondisi partisi dan arm SMOTE.

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha=0,5$*],
        table.cell(colspan: 2)[*IID*],
        [none], [SMOTE], [none], [SMOTE], [none], [SMOTE],
      ),
      [LR], [0,750], [0,767], [0,757], [0,772], [0,758], [0,768],
      [SVM], [0,784], [0,789], [0,743], [0,742], [0,741], [0,734],
      [GBM], [0,761], [0,833], [0,726], [0,827], [0,698], [0,811],
      [FFD], [0,788], [0,794], [0,814], [0,825], [0,802], [0,809],
      [BERT], [0,774], [0,796], [0,779], [0,804], [0,818], [0,823],
      [FedXGBllr], [—], [—], [0,724], [0,805], [0,712], [0,806],
      [XGBoost], [0,838], [0,831], [—], [—], [—], [—],
    )
  ],
  caption: [AUPRC test pada ULB],
) <tab-4-auprc-ulb>

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      table.header(
        [*Model*], [*Centralized (none)*], [*Dirichlet none*],
        [*Dirichlet SMOTE*], [*IID (none)*],
      ),
      [LR], [0,144], [0,139], [0,097], [0,144],
      [SVM], [0,144], [0,119], [0,081], [0,085],
      [GBM], [0,161], [0,162], [0,162], [0,137],
      [FFD], [0,159], [0,157], [0,045], [0,158],
      [BERT], [0,169], [0,167], [0,045], [0,167],
      [FedXGBllr], [—], [0,151], [0,137], [0,141],
      [XGBoost], [0,157], [—], [—], [—],
    )
  ],
  caption: [AUPRC test pada BAF],
) <tab-4-auprc-baf>

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha=0,5$*],
        table.cell(colspan: 2)[*IID*],
        [none], [SMOTE], [none], [SMOTE], [none], [SMOTE],
      ),
      [LR], [0,603], [0,624], [0,601], [0,595], [0,612], [0,656],
      [SVM], [0,605], [0,596], [0,577], [0,572], [0,311], [0,632],
      [GBM], [0,996], [0,997], [0,996], [0,996], [0,995], [0,996],
      [FFD], [0,745], [0,825], [0,647], [0,678], [0,756], [0,839],
      [BERT], [0,830], [0,938], [0,660], [0,641], [0,858], [0,916],
      [FedXGBllr], [—], [—], [0,996], [0,995], [0,996], [0,996],
      [XGBoost], [0,985], [0,985], [—], [—], [—], [—],
    )
  ],
  caption: [AUPRC test pada PaySim],
) <tab-4-auprc-paysim>

Kesukaran ketiga dataset berbeda tajam apabila AUPRC dinyatakan sebagai kelipatan
baseline acak masing-masing. Nilai terbaik tiap dataset setara dengan sekitar 15
kali baseline pada BAF, 484 kali pada ULB, dan 773 kali pada PaySim. Perbandingan
ini menegaskan bahwa BAF secara intrinsik jauh lebih sukar sehingga angka 0,16
pada BAF bukan pertanda kegagalan model melainkan cerminan bahwa sinyal fraud di
sel tersebut memang lebih lemah relatif terhadap baseline yang sudah tinggi. Seluruh
pembacaan lintas dataset pada bab ini karena itu dilakukan terhadap baseline
masing-masing, bukan terhadap angka AUPRC mentah.

Penurunan performa yang ditimbulkan paradigma federated ternyata bergantung pada
dataset dan keluarga model dan bukan akibat pengaruh tunggal. Pada ULB dan
PaySim, model tree berupa GBM dan FedXGBllr nyaris tidak kehilangan performa antara
kondisi terpusat dan federated dimana FedXGBllr pada PaySim bertahan di sekitar 0,996
pada seluruh kondisi sedangkan GBM pada ULB justru naik dari kondisi IID ke
kondisi terpusat pada arm SMOTE. Sebaliknya, model deep learning memperlihatkan gambaran yang
berlawanan, dengan penurunan yang jelas di bawah partisi Dirichlet dimana BERT pada
PaySim turun dari 0,858 pada kondisi IID menjadi 0,660 pada kondisi Dirichlet untuk
arm tanpa SMOTE. Maka dari itu, penurunan performa federasi lebih tepat dipahami sebagai hasil
interaksi antara tingkat heterogenitas partisi dan kerentanan keluarga model dan bukan
sebagai ongkos tetap yang melekat pada paradigma federated itu sendiri.

AUPRC mengukur kualitas peringkat pada seluruh rentang ambang, sedangkan
Recall\@5%FPR membaca performa pada satu titik operasi yang dapat diterapkan,
yaitu proporsi fraud yang tertangkap ketika false positive rate dibatasi pada 5
persen. @tab-4-rfpr-ulb, @tab-4-rfpr-baf, dan @tab-4-rfpr-paysim menyajikan metrik
tersebut dalam tata letak yang sama dengan tabel AUPRC sebelumnya, dihitung dari
skor model yang telah dibekukan.

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha=0,5$*],
        table.cell(colspan: 2)[*IID*],
        [none], [SMOTE], [none], [SMOTE], [none], [SMOTE],
      ),
      [LR], [0,878], [0,878], [0,878], [0,878], [0,878], [0,865],
      [SVM], [0,878], [0,878], [0,892], [0,892], [0,892], [0,892],
      [GBM], [0,851], [0,865], [0,824], [0,865], [0,770], [0,905],
      [FFD], [0,865], [0,878], [0,865], [0,919], [0,865], [0,892],
      [BERT], [0,878], [0,878], [0,892], [0,878], [0,851], [0,905],
      [FedXGBllr], [—], [—], [0,851], [0,878], [0,878], [0,878],
      [XGBoost], [0,892], [0,892], [—], [—], [—], [—],
    )
  ],
  caption: [Recall\@5%FPR test pada ULB],
) <tab-4-rfpr-ulb>

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      table.header(
        [*Model*], [*Centralized (none)*], [*Dirichlet none*],
        [*Dirichlet SMOTE*], [*IID (none)*],
      ),
      [LR], [0,528], [0,521], [0,430], [0,527],
      [SVM], [0,524], [0,477], [0,389], [0,404],
      [GBM], [0,563], [0,560], [0,560], [0,519],
      [FFD], [0,545], [0,550], [0,253], [0,548],
      [BERT], [0,564], [0,574], [0,271], [0,570],
      [FedXGBllr], [—], [0,537], [0,506], [0,525],
      [XGBoost], [0,545], [—], [—], [—],
    )
  ],
  caption: [Recall\@5%FPR test pada BAF],
) <tab-4-rfpr-baf>

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha=0,5$*],
        table.cell(colspan: 2)[*IID*],
        [none], [SMOTE], [none], [SMOTE], [none], [SMOTE],
      ),
      [LR], [0,919], [0,951], [0,933], [0,923], [0,922], [0,957],
      [SVM], [0,932], [0,941], [0,907], [0,915], [0,571], [0,938],
      [GBM], [0,999], [0,998], [0,999], [0,996], [0,996], [0,997],
      [FFD], [0,944], [0,976], [0,886], [0,925], [0,953], [0,976],
      [BERT], [0,997], [0,997], [0,972], [0,861], [0,996], [0,997],
      [FedXGBllr], [—], [—], [0,996], [0,996], [0,996], [0,996],
      [XGBoost], [0,996], [1,000], [—], [—], [—], [—],
    )
  ],
  caption: [Recall\@5%FPR test pada PaySim],
) <tab-4-rfpr-paysim>

Kedua metrik menghasilkan penilaian yang berbeda terhadap eksperimen dengan dataset BAF. 
Dibaca lewat AUPRC, BAF tampak nyaris gagal dengan skor terbaik hanya sekitar 0,16.
Namun pada titik operasi 5 persen FPR, model-model BAF menangkap proporsi fraud
yang bermakna secara operasional dengan kira-kira 52 hingga
57 persen fraud tertangkap sambil hanya menandai 5 persen transaksi sah. 
SVM pada federated dan model deep learning di bawah partisi Dirichlet arm SMOTE
menjadi pengecualian dengan capaian lebih lemah. Maka dari itu, meskipun 
BAF lemah pada presisi peringkat namun masih memadai pada deteksi di
titik operasi yang layak pakai dalam hal fraud detection.

Meskipun secara umum kedua metrik sepakat dalam menilai performa tiap-tiap sel, 
ada beberapa ketidaksepakatan yang muncul. Ketidaksepakatan terbesar justru muncul 
pada sel dimana peringkat AUPRC runtuh tetapi kemampuan deteksi bertahan.
Contoh paling tajam adalah SVM pada PaySim kondisi IID tanpa SMOTE, yang AUPRC-nya hanya 0,311
sedangkan Recall\@5%FPR-nya mencapai 0,571 dan berarti model tetap memulihkan
sekitar 57 persen fraud pada anggaran 5 persen FPR meski peringkat keseluruhannya
buruk. Pola serupa juga terjadi pada seluruh ULB dimana AUPRC berkisar antara 0,70 dan 0,84
sementara Recall\@5%FPR konsisten lebih tinggi pada rentang 0,85 hingga 0,92. Hal ini disebabkan
karena fitur PCA ULB menyediakan wilayah recall-tinggi yang bersih meski presisi
peringkat keseluruhannya hanya sedang. Sebuah model dapat berperingkat baik secara
menyeluruh namun buruk pada wilayah presisi-tinggi yang ditimbang berat oleh AUPRC,
atau justru sebaliknya, dan karena itu kedua metrik dilaporkan berdampingan.

=== Performa antar paradigma agregasi

Perbandingan performa antar keempat paradigma agregasi tidak menghasilkan satu
urutan tunggal yang berlaku lintas dataset, melainkan urutan yang bergantung pada
interaksi antar dataset dan kondisi eksperimen. Keunggulan model tree atas model 
parametrik dengan agregasi FedAvg terlihat paling tegas pada dataset dengan 
struktur transaksional yang relatif lebih rumit. Pada PaySim, FedXGBllr
mempertahankan AUPRC sekitar 0,996 di seluruh kondisi sementara LR bergerak pada
rentang 0,595 hingga 0,656 dan SVM pada rentang 0,311 hingga 0,632. Pada BAF Dirichlet, 
FedXGBllr sebesar 0,151 juga melampaui LR sebesar 0,139 dan SVM sebesar 0,119 
sedangkan pada ULB keunggulan itu menyempit dan bahkan berbalik pada arm tanpa SMOTE 
dimana FedXGBllr hanya mencapai antara 0,712 dan 0,724 berbanding LR antara 0,757 dan 0,758. 
Pola ini mencerminkan bahwa keunggulan model tree pada data tabular yang dilaporkan 
#cite(<grinsztajn2022>, form: "prose") bertahan ketika model dipindahkan ke ekosistem federated, 
tetapi bergantung pada tersedianya interaksi non-linear yang dapat dieksploitasi oleh struktur pohon.

Model deep dengan agregasi FedAvg terbobot-akurasi menunjukkan hasil dengan langit-langit tertinggi 
namun disertai lantai terendah. Pencapaian tertinggi seluruh studi pada BAF diraih BERT sebesar 0,1670 yang
melampaui FT-Transformer terpusat terpublikasi sebesar 0,1607 dan capaian itu
diperoleh justru pada partisi Dirichlet yang merupakan kondisi paling heterogen dalam matriks. 
Namun paradigma yang sama memperlihatkan sensitivitas partisi terbesar sebab BERT pada PaySim menurun dari
0,858 pada kondisi IID menjadi 0,660 pada kondisi Dirichlet, penurunan yang tidak
dialami model tree mana pun. Penurunan performa akibat federasi dengan demikian bukan besaran tunggal
milik paradigma federated, melainkan hasil interaksi antara heterogenitas partisi
dan kerentanan keluarga model, sejalan dengan temuan #cite(<li2021noniid>, form: "prose")
bahwa degradasi akibat Non-IID terkonsentrasi pada algoritma agregasi yang
mengasumsikan keseragaman distribusi.

Secara keseluruhan, perbandingan antar paradigma agregasi dapat disimpulkan sebagai
berikut. FedXGBllr dengan agregasi tree ensemble memberikan performa yang setara
dengan GBM ber-best-model selection,
unggul atas LR dan SVM ber-FedAvg pada data dengan struktur non-linear yang kaya,
dan sedikit di bawah model deep ber-FedAvg terbobot-akurasi pada puncaknya namun
dengan kestabilan lintas kondisi partisi yang jauh lebih baik daripada keduanya.
Tidak satu pun paradigma unggul secara serentak pada performa di kedua metrik dan ketahanan
terhadap heterogenitas sebab setiap paradigma menukar satu properti
dengan properti lainnya. Temuan ini menegaskan bahwa pemilihan paradigma agregasi
dalam deteksi fraud kolaboratif lintas institusi merupakan keputusan bersyarat yang
bergantung pada karakteristik separabilitas data, tingkat heterogenitas antar
institusi, dan apakah keluaran model akan dipakai sebagai peringkat atau sebagai
estimasi probabilitas.

=== Analisis lanjutan pada eksperimen dengan dataset BAF

Nilai AUPRC BAF di sekitar 0,16 tampak seperti kegagalan bila disandingkan dengan
ULB yang mencapai sekitar 0,83, padahal rentang itulah yang dihasilkan BAF Base
bagi semua pendekatan, termasuk state of the art terpusat yang telah
dipublikasikan. #cite(<dong2026fcorr>, form: "prose") melaporkan AUPRC test pada
BAF Base dengan dataset, varian, dan metrik yang sama, yaitu 1.000.000 sampel pada
prevalensi sekitar 1,1 persen, untuk sejumlah arsitektur terpusat sebagaimana
disajikan pada @tab-4-baf-auprc-bench; model terkuat mereka, FT-Transformer,
mencapai 0,1607.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto),
      align: (left, right, left),
      table.header([*Model*], [*AUPRC*], [*Setting*]),
      [TabTransformer @dong2026fcorr], [0,1080], [terpusat],
      [FFN @dong2026fcorr], [0,1234], [terpusat],
      [LightGBM @dong2026fcorr], [0,1442], [terpusat],
      [FCorrTransformer @dong2026fcorr], [0,1458], [terpusat],
      [FT-Transformer + CAR @dong2026fcorr], [0,1602], [terpusat],
      [FT-Transformer @dong2026fcorr], [0,1607], [terpusat],
      table.hline(),
      [BERT (studi ini)], [*0,1670*], [federated, Dirichlet $alpha = 0,5$],
      [GBM (studi ini)], [0,1620], [federated, Dirichlet $alpha = 0,5$],
      [FFD (studi ini)], [0,1581], [federated, IID],
      [FedXGBllr (studi ini)], [0,1515], [federated, Dirichlet $alpha = 0,5$],
      [LR (studi ini)], [0,1440], [federated, IID],
      [SVM (studi ini)], [0,1194], [federated, Dirichlet $alpha = 0,5$],
      [XGBoost (studi ini)], [0,1569], [n/a (terpusat)],
    )
  ],
  caption: [Perbandingan AUPRC test pada BAF Base],
) <tab-4-baf-auprc-bench>

Kolom studi ini pada tabel tersebut memuat hasil terbaik federated per model,
sedangkan sel XGBoost merupakan baseline terpusat dan ditandai sebagai tidak
berlaku. BERT mencapai 0,1670 sehingga melampaui FT-Transformer terpublikasi yang
mencapai 0,1607, dan perlu dicatat bahwa model BERT pada studi ini berasal dari
keluarga arsitektur yang sama. GBM dengan 0,1620 juga melampauinya, sementara LR
dengan 0,1440 praktis setara dengan LightGBM terpusat mereka yang mencapai 0,1442.
Yang lebih penting, kedua capaian terbaik BERT, yaitu AUPRC 0,1670 dan
Recall\@5%FPR 0,5738, berasal dari sel Dirichlet dengan parameter konsentrasi 0,5
yang merupakan kondisi paling heterogen dalam matriks eksperimen, bukan dari
kondisi IID. Hal ini memperkuat klaim bahwa performa federated pada studi ini
tidak bergantung pada partisi yang mudah.

Pada titik operasi tetap, #cite(<nasif2026csnpc>, form: "prose") menghimpun hasil
BAF terpublikasi di sekitar 5 persen FPR sebagaimana disajikan pada
@tab-4-baf-recall-bench. Capaian Recall\@5%FPR terbaik studi ini, yaitu BERT
sebesar 0,5738 pada Dirichlet dengan parameter konsentrasi 0,5, melampaui seluruh
entri terpublikasi termasuk SpikeConv M5 yang mencapai 0,570#footnote[Angka utama
90,8% pada #cite(<nasif2026csnpc>, form: "prose") bukan hasil pada Base: Tabel 3
mereka memberi P200-S20 pada Base sebesar TPR 0,476 pada FPR 0,014, sedangkan 0,908
merujuk varian lain — teks mereka menyebut Variant II sementara tabel mereka
menunjukkan Variant I, sebuah inkonsistensi internal. Nilai Base-komparabel 0,476
yang dipakai di sini.], dan capaian itu diperoleh dalam setting federated.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto),
      align: (left, left, right),
      table.header([*Sumber*], [*Model*], [*Recall\@5%FPR*]),
      [Ribeiro dkk. 2025 @ribeiro2025spikeconv], [SpikeConv M5], [0,570],
      [Luzio dkk. 2024 @luzio2024calibration], [LightGBM], [0,540],
      [Luzio dkk. 2024 @luzio2024calibration], [CatBoost], [0,520],
      [Luzio dkk. 2024 @luzio2024calibration], [MLP], [0,490],
      [Uwaoma 2024 (tesis)], [Random Forest], [0,480],
      [Nasif dkk. 2026 @nasif2026csnpc], [CSNPC+RHOSS (Base)], [0,476],
      [Perdigão dkk. 2024 @perdigao2024snn], [CSNN], [0,471],
      [Uwaoma 2024 (tesis)], [LightGBM], [0,470],
      [Ribeiro dkk. 2025 @ribeiro2025spikeconv], [LightGBM (GBDT)], [0,450],
      [Pombal dkk. 2022 @pombal2022unfairness], [6 model klasik], [0,25–0,75],
      table.hline(),
      [Studi ini], [BERT, Dirichlet $alpha = 0,5$], [*0,5738*],
      [Studi ini], [GBM], [0,5605],
      [Studi ini], [FFD], [0,5496],
      [Studi ini], [FedXGBllr], [0,5369],
      [Studi ini], [LR], [0,5272],
      [Studi ini], [SVM], [0,4770],
      [Studi ini], [XGBoost (terpusat)], [0,5453],
    )
  ],
  caption: [Perbandingan Recall\@5%FPR pada BAF Base],
) <tab-4-baf-recall-bench>

Tiga peringatan perlu dinyatakan terus terang agar perbandingan tersebut tidak
menyesatkan. Pertama dan paling penting, protokol pemisahan data berbeda: studi
ini memakai split acak terstratifikasi dengan proporsi 70:15:15, sedangkan
protokol standar BAF menurut #cite(<jesus2022baf>, form: "prose") bersifat temporal
dengan bulan pertama hingga keenam untuk pelatihan dan bulan ketujuh hingga
kedelapan untuk pengujian. Split acak tidak menuntut generalisasi lintas waktu
sehingga merupakan setting yang lebih mudah, dan tanpa peringatan ini perbandingan
menjadi tidak adil. Kedua, titik operasinya berbeda,
karena #cite(<perdigao2024snn>, form: "prose") melaporkan pada FPR 4,32 persen
dan #cite(<nasif2026csnpc>, form: "prose") pada 1,4 persen untuk varian Base,
sedangkan studi ini pada sekitar 5 persen; recall pada FPR yang lebih rendah
merupakan target yang lebih sukar. Ketiga, sebagian sumber pembanding bersifat
non-arsip,
yaitu #cite(<pombal2022unfairness>, form: "prose") yang melaporkan rentang lintas
varian dan Uwaoma (2024) yang berupa tesis magister.

Simpulan dari perbandingan ini adalah bahwa AUPRC absolut BAF di sekitar 0,16 bukan
bukti kegagalan model. Hasil terpusat terpublikasi pada dataset dan varian yang
sama berada pada rentang yang sama, dengan capaian terkuat 0,1607 untuk
FT-Transformer. Model federated pada studi ini mencapai nilai yang setara atau
lebih tinggi, dan capaian Recall\@5%FPR terbaiknya melampaui seluruh hasil
terpublikasi yang dihimpun. BAF memang sukar secara intrinsik, dan sub-subbab
berikut mengkuantifikasi penyebabnya.

Pengamatan ini tampak paradoks. Prevalensi fraud BAF sebesar 8,5 kali prevalensi ULB, 
yaitu 1,10 persen berbanding 0,172 persen, namun BAF hanya
mencapai sekitar 15 kali baseline sementara ULB mencapai sekitar 484 kali.
Seandainya ketidakseimbangan kelas merupakan faktor kesukaran yang dominan, urutan
tersebut seharusnya terbalik. Hipotesis yang diajukan adalah bahwa separabilitas
kelas, dan bukan ketidakseimbangan, yang mengendalikan perbedaan itu, dengan
dugaan bahwa kelas minoritas BAF jauh lebih menyatu dengan kelas mayoritas daripada
kelas minoritas ULB. Analisis berikut menguji hipotesis tersebut secara kuantitatif
dan bersifat baca-saja atas data terpraproses yang telah di-cache, tanpa melatih
satu model pun.

Bukti utamanya adalah tipologi contoh minoritas mengikuti
#cite(<napierala2016types>, form: "prose"). Setiap contoh fraud pada data latih
diklasifikasikan menurut komposisi lima tetangga terdekatnya, dengan jarak
Euclidean pada fitur terskala, menjadi *safe* apabila memiliki empat hingga lima
tetangga minoritas, *borderline* untuk dua hingga tiga tetangga, *rare* untuk satu
tetangga, dan *outlier* apabila tidak memiliki tetangga minoritas sama sekali.
Referensi tetangga bagi ULB dan BAF berupa seluruh data latih, sedangkan bagi
PaySim berupa subsampel uniform berukuran 500.000 sampel dengan seed 42 yang
mempertahankan prevalensi aslinya. @fig-typology dan @tab-typology menyajikan
hasilnya.

#figure(
  image("resources/fig-4-2-minority-typology.png", width: 92%),
  caption: [Distribusi tipe contoh minoritas (k = 5 tetangga terdekat) per dataset.
  Kelas minoritas BAF hampir seluruhnya *rare* dan *outlier*, sedangkan ULB
  didominasi *safe*.],
) <fig-typology>

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header([*Dataset*], [*dim*], [*safe*], [*borderline*], [*rare*],
        [*outlier*], [*rare+outlier*]),
      [ULB], [30], [71,5%], [9,9%], [2,3%], [16,3%], [*18,6%*],
      [BAF], [55], [0,3%], [6,6%], [19,3%], [73,8%], [*93,1%*],
      [PaySim], [13], [57,6%], [10,7%], [7,8%], [24,0%], [*31,8%*],
    )
  ],
  caption: [Tipologi contoh minoritas],
) <tab-typology>

Kelas minoritas BAF terdiri atas 93,1 persen contoh *rare* dan *outlier*, dengan
73,8 persen di antaranya berupa *outlier* murni tanpa satu pun tetangga minoritas,
sedangkan kelas minoritas ULB terdiri atas 81,4 persen contoh *safe*
dan *borderline*. Mengikuti #cite(<napierala2016types>, form: "prose"), kelas
minoritas yang didominasi contoh *rare* dan *outlier* tidak dapat dipelajari secara
andal berapa pun banyaknya contoh semacam itu, karena contoh-contoh tersebut tidak
membawa struktur lokal yang dapat digeneralisasi oleh classifier. Temuan ini
merupakan wujud pembedaan antara kelangkaan absolut dan kelangkaan relatif menurut
#cite(<weiss2004rarity>, form: "prose"): BAF memiliki lebih banyak kasus fraud,
namun kasus-kasus tersebut tersebar di wilayah mayoritas.

== Pengaruh Heterogenitas Distribusi Data dan Penanganan Class Imbalance

=== Pengaruh heterogenitas distribusi tanpa oversampling

Pengaruh heterogenitas distribusi perlu dibaca terlebih dahulu secara terisolasi,
yaitu pada arm tanpa SMOTE, agar efeknya tidak tercampur dengan efek oversampling
yang dibahas setelahnya. Perbandingan kolom IID dan kolom Dirichlet pada
@tab-4-auprc-ulb, @tab-4-auprc-baf, dan @tab-4-auprc-paysim mengungkap bahwa
heterogenitas partisi tidak menurunkan performa secara seragam, dan pada sebagian
sel bahkan tidak menurunkannya sama sekali.

Pada dataset BAF, perpindahan dari partisi IID ke partisi Dirichlet justru menaikkan AUPRC
bagi mayoritas model. Pola serupa terlihat pada dataset ULB dimana penurunan hanya
dialami BERT. Temuan ini melawan intuisi umum bahwa Non-IID selalu merugikan namun kenyataannya
partisi Dirichlet mengonsentrasikan kelas minoritas pada sebagian client alih-alih
menyebarkannya ke seluruh client sehingga client yang menerima konsentrasi tersebut
justru memperoleh sinyal minoritas yang lebih padat daripada yang tersedia di bawah
pembagian seragam.

Di lain sisi, kerugian akibat heterogenitas terkonsentrasi pada keluarga model deep learning
terutama pada dataset PaySim. FFD menurun dari 0,756 pada kondisi IID menjadi 0,647 pada
kondisi Dirichlet, sedangkan BERT menurun lebih dalam dari 0,858 menjadi 0,660.
Heterogenitas distribusi dengan demikian bukan faktor perusak yang berdiri sendiri.
Pengaruhnya bergantung pada keluarga model dan arahnya bahkan dapat positif ketika
konsentrasi minoritas menguntungkan sebagian client. Pengamatan ini menjadi latar
penting bagi seluruh sub-subbab berikutnya karena penurunan performa tidak hanya muncul
akibat heterogenitas melainkan dari interaksinya dengan oversampling lokal.

=== Pengaruh penanganan class imbalance dengan SMOTE

Efek SMOTE tidak seragam, melainkan bergantung pada apakah oversampling yang terjadi
bersifat moderat dan menyeluruh ataukah ekstrem dan terkonsentrasi.
Generalisasi yang dapat ditarik adalah bahwa oversampling moderat pada seluruh
client cenderung membantu sedangkan oversampling ekstrem pada segelintir client
yang kelaparan minoritas meracuni agregat global. Hal ini ditentukan oleh interaksi antar
partisi, dataset, dan model serta agregasinya.

Untuk menunjukkan hal ini, @tab-4-baf-smote menyajikan efek SMOTE pada BAF dengan partisi Dirichlet
dipilah menurut paradigma agregasi. Seluruh baris berbagi dataset, partisi, seed, dan client teracuni yang sama
sehingga satu-satunya yang berbeda adalah aturan agregasinya.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, left, right, right, right),
      table.header([*Paradigma agregasi*], [*Model*], [*no-SMOTE*],
        [*SMOTE*], [*$Delta$*]),
      [Best-model selection], [GBM], [0,162], [0,162], [0,0%],
      [Tree ensemble aggregation], [FedXGBllr], [0,151], [0,137], [−9,3%],
      [FedAvg], [LR], [0,139], [0,097], [−30,5%],
      [FedAvg], [SVM], [0,119], [0,081], [−31,9%],
      [Accuracy-weighted FedAvg], [FFD], [0,157], [0,045], [−71,0%],
      [Accuracy-weighted FedAvg], [BERT], [0,167], [0,045], [−73,2%],
    )
  ],
  caption: [Efek SMOTE terhadap AUPRC pada BAF Dirichlet],
) <tab-4-baf-smote>

Agar temuan tersebut tidak bergantung pada satu metrik, @tab-4-baf-smote-rfpr
menghitung ulang efek yang sama dalam Recall\@5%FPR.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, left, right, right, right),
      table.header([*Paradigma agregasi*], [*Model*], [*no-SMOTE*],
        [*SMOTE*], [*$Delta$*]),
      [Best-model selection], [GBM], [0,560], [0,560], [0,0%],
      [Tree ensemble aggregation], [FedXGBllr], [0,537], [0,506], [−5,7%],
      [FedAvg], [LR], [0,521], [0,430], [−17,3%],
      [FedAvg], [SVM], [0,477], [0,389], [−18,5%],
      [Accuracy-weighted FedAvg], [FFD], [0,550], [0,253], [−53,9%],
      [Accuracy-weighted FedAvg], [BERT], [0,574], [0,271], [−52,8%],
    )
  ],
  caption: [Efek SMOTE terhadap Recall\@5%FPR pada BAF Dirichlet],
) <tab-4-baf-smote-rfpr>

Peringkat ketahanan antar paradigma tidak berubah ketika metriknya diganti.
Best-model selection tetap tidak tersentuh dengan penurunan nol persen, tree
ensemble aggregation mengalami penurunan ringan sebesar 5,7 persen, kedua model
parametrik dengan FedAvg berada di tengah dengan penurunan 17,3 persen dan 18,5
persen, sedangkan FedAvg terbobot-akurasi pada model deep runtuh paling dalam
sebesar 53,9 persen dan 52,8 persen. Kemunculan urutan monoton yang sama pada
metrik operasional yang sepenuhnya berbeda menjadikan menunjukkan klaim yang kuat
bahwa efek SMOTE bergantung pada interaksi keseluruhan variabel eksperimen dimana efek terburuk
terjadi ketika kondisi partisi menyebabkan oversampling ekstrem pada client-client tertentu.
Kemudian dengan agregasi bersifat Accuracy-weighted FedAvg, clien-client ini diberikan bobot kuat
dalam model global sehingga memberikan hasil yang menyesatkan.

=== Analisis lanjutan pengaruh heterogenitas dan SMOTE

Kedua variabel yang dikaji pada subbab ini ternyata tidak setara bobotnya, dan
keduanya tidak bekerja secara aditif. Heterogenitas distribusi yang dibaca
sendirian, yaitu pada arm tanpa SMOTE, tidak menurunkan performa secara sistematis.
Kerugian akibat heterogenitas terkonsentrasi pada keluarga model deep, 
terutama pada dataset PaySim. SMOTE yang dibaca sendirian juga tidak
seragam arahnya sebab tergantung pada kondisi oversampling yang terjadi.
Yang menentukan arah bukanlah salah satu variabel,
melainkan interaksi keduanya.

Interaksi itu bekerja melalui satu jalur yang dapat ditelusuri. Partisi Dirichlet
menciptakan client yang kelaparan minoritas, SMOTE lokal pada client semacam itu
menghasilkan sampel sintetis berbentuk wireframe alih-alih awan, model yang dipaskan
pada wireframe tersebut melaporkan metrik lokal yang nyaris sempurna, dan aturan
agregasi yang memercayai metrik yang dilaporkan sendiri itu kemudian menyerahkan
mayoritas bobot kepada client yang paling rusak. Tidak satu pun mata rantai tersebut
berbahaya secara terpisah dan kerusakan baru muncul ketika keempatnya tersambung.

Konsekuensinya dapat dinyatakan sebagai ketergantungan berjenjang. Besarnya
kerusakan tidak ditentukan oleh tingkat heterogenitas maupun oleh penerapan SMOTE,
melainkan oleh aturan agregasi yang menjembatani keduanya, dan selisih antar aturan
itu membentang dari nol persen hingga 73,2 persen pada kondisi yang identik.
Bagi rancangan sistem deteksi fraud
kolaboratif, temuan ini menyiratkan bahwa penanganan class imbalance pada lingkungan
federated tidak dapat diputuskan terlepas dari aturan agregasi yang dipakai, sebab
intervensi yang menguntungkan pada partisi seragam dapat berbalik menjadi merusak
pada partisi heterogen dengan aturan pembobotan yang keliru.

== Interpretabilitas Model <sec-hasil-rq3>

Analisis explainability dijalankan terhadap model global akhir yang telah
dibekukan dan dipersistensi oleh sweep, sehingga SHAP berperan sebagai konsumen
baca-saja atas artefak tersebut tanpa melatih ulang apa pun. Tahap utama mencakup
96 sel, dengan 66 di antaranya berupa sel federated yang memiliki lebih dari satu
client. Hanya sel federated yang membawa informasi stabilitas antar client,
sedangkan 30 sel centralized hanya memiliki satu client sehingga stabilitasnya
tidak terdefinisi menurut definisi dan dikeluarkan dari seluruh agregat pada subbab
ini. Di luar tahap utama dijalankan dua tahap tambahan, yaitu 12 sel arm background
bersama dan 16 sel tingkat eksak PaySim, sehingga berkas ringkasan
`results/shap_v2/shap_summary_v2.csv` memuat 124 baris sel. Seluruh 124 sel
berstatus `ok`; tidak ada sel yang gugur karena degenerasi pada pengukuran ini.

=== Konfigurasi pengukuran

Pada setiap client, SHAP dihitung terhadap model global akhir menggunakan
background berupa 100 sampel data latih lokal pasca-SMOTE yang diringkas menjadi 10
sentroid k-means. Ringkasan tersebut menentukan distribusi referensi explainer.
Explanation set berupa 500 sampel dari test set terpusat yang identik untuk seluruh
client. Seluruh atribusi dihitung pada skala log-odds, dengan margin fungsi
keputusan sebagai pengecualian terdokumentasi untuk SVM.

Konfigurasi tersebut menentukan makna klaim pada arm utama. Model yang dijelaskan
bersifat global dan identik bagi seluruh client, dan baris yang dijelaskan pun
identik bagi seluruh client; satu-satunya masukan yang bervariasi antar client
adalah distribusi referensi berupa background lokalnya. Divergensi yang terukur
pada arm utama karena itu bukan pernyataan umum bahwa client berselisih tentang
model, melainkan pernyataan yang lebih sempit sekaligus lebih presisi: dengan model
dan baris yang diaudit ditahan identik, mengganti distribusi referensi lokal saja
sudah mengubah fitur yang tampak diandalkan model.

Khusus untuk KernelSHAP, dua penetapan konfigurasi menentukan validitas seluruh
angka pada subbab ini. Pertama, seleksi fitur bawaan dinonaktifkan melalui
`l1_reg=False`. Sejak versi 0.47.0 pustaka SHAP menetapkan `num_features(10)`
sebagai nilai bawaan, sehingga LARS hanya mempertahankan sepuluh fitur dan
memberikan nilai nol eksak pada seluruh fitur lain untuk setiap baris yang
dijelaskan; pada BAF dengan 55 fitur hal itu berarti sekurang-kurangnya 45 nol
eksak per baris. Karena seleksi LARS merupakan fungsi yang tidak kontinu terhadap
undian koalisi, perilaku bawaan tersebut menaikkan lantai derau jauh lebih besar
daripada penyamplingannya sendiri. Kedua, setiap client dijelaskan pada dua seed
koalisi khusus SHAP, yaitu 11 dan 22, dengan explanation set dan background yang
identik antar seed. Seluruh perbandingan antar client memasangkan vektor yang
dihitung di bawah seed koalisi yang sama, yakni skema common random numbers,
sehingga galat estimator sebagian besar saling meniadakan pada kontras antar
client.

Pemetaan explainer ditetapkan berdasarkan pengukuran local accuracy alih-alih
berdasarkan reputasi masing-masing metode, sebagaimana disajikan pada
@tab-4-shap-explainer. Nilai local accuracy diambil dari probe Tahap-0 yang terekam
pada `results/shap_stage0_report.txt`.

#figure(
  table(
    columns: 3,
    align: (left, left, center),
    table.header([*Model*], [*Explainer*], [*Local accuracy*]),
    [LR, SVM], [LinearSHAP (SVM: margin)], [$9,26 times 10^(-8)$],
    [GBM, XGB], [TreeSHAP `interventional`], [$0,00$],
    [FFD, BERT, FedXGBllr], [KernelSHAP ($"nsamples" = 500$, `l1_reg=False`)], [—],
  ),
  kind: table,
  caption: [Pemetaan explainer per model],
) <tab-4-shap-explainer>

Pemilihan mode interventional alih-alih `tree_path_dependent` untuk model pohon
merupakan keputusan yang menentukan apakah keluarga model tersebut dapat ikut
dianalisis sama sekali. Mode `tree_path_dependent` mengabaikan data background,
sehingga setiap client yang menjelaskan satu model global yang sama menghasilkan
nilai yang identik dan stabilitas antar client bernilai 1,0 secara konstruksi;
angka semacam itu akan mengeluarkan model pohon dari analisis ini sepenuhnya.
Mode interventional dengan background lokal per client menjadikan model pohon
peserta yang sesungguhnya dalam perbandingan. Asumsi independensi fitur yang
dituntutnya merupakan asumsi yang sama yang telah didokumentasikan untuk
LinearSHAP dan KernelSHAP, sehingga pemilihan ini tidak memperkenalkan kelas
batasan baru.

=== Admisibilitas explainer per keluarga model

Perbedaan pertama antar keluarga model bukan terletak pada seberapa stabil
interpretasinya, melainkan pada apakah interpretasi yang eksak dimungkinkan sama
sekali. Model parametrik menerima LinearSHAP dengan galat local accuracy sebesar
9,26e−8, sedangkan model pohon murni menerima TreeSHAP interventional dengan galat
tepat nol. Keduanya karena itu memenuhi aksioma local accuracy secara eksak, dan
atribusinya mendekomposisi prediksi tanpa sisa.

Model deep tidak memiliki estimator eksak yang berlaku, dan hal itu merupakan
properti arsitekturnya alih-alih keterbatasan anggaran komputasi. DeepSHAP diuji
lebih dahulu karena sifatnya yang mendekati eksak, dan pada FFD yang berbasis
1D-CNN metode tersebut memenuhi aksioma local accuracy dengan galat rekonstruksi
sebesar 1,37e−6. Pada BERT yang berbasis FT-Transformer, DeepSHAP gagal karena
pustaka SHAP menaikkan AssertionError bahwa atribusi tidak menjumlah ke luaran
model. Penyebabnya adalah lapisan LayerNorm yang hadir pada setiap blok Transformer,
dan DeepLIFT tidak memiliki aturan propagasi untuk lapisan tersebut.
GradientExplainer yang diuji sebagai alternatif melaporkan galat local accuracy
sebesar 13,6 terhadap toleransi 0,01, yakni sekitar 1.360 kali di atas ambang,
sehingga atribusinya tidak mendekomposisi prediksi dan dicatat sebagai estimator
yang gagal alih-alih sebagai pendekatan. BERT karena itu memakai KernelSHAP.
Meskipun DeepSHAP lolos pada FFD, FFD tetap memakai KernelSHAP demi komparabilitas,
sebab apabila FFD memakai estimator yang nyaris eksak sementara BERT memakai
estimator berbasis sampling, ketidakstabilan BERT yang teramati akan sebagian
mencerminkan varians estimator alih-alih perilaku model, padahal kedua model deep
tersebut dibandingkan secara langsung. Run DeepSHAP pada FFD dipertahankan sebagai
pemeriksaan silang.

FedXGBllr menempati posisi yang tidak terduga dalam pemetaan ini, dan posisinya
merupakan temuan tersendiri. Meskipun berbasis tree ensemble, model ini
tidak mewarisi keeksakan TreeSHAP yang dinikmati GBM, karena kepala agregasinya
berupa jaringan konvolusi satu dimensi yang memuat aktivasi ReLU sehingga linearitas
Shapley terputus, sebagaimana diuraikan pada Subbab Implementasi Modul Evaluasi dan
SHAP. Konsekuensinya, FedXGBllr harus dijelaskan dengan KernelSHAP yang bersifat
model-agnostik dan berbasis sampling, sejajar dengan kedua model deep. Kategori
"model berbasis tree" karena itu tidak menentukan karakteristik explainability
secara otomatis; yang menentukan adalah keseluruhan jalur komputasi dari masukan
hingga keluaran, termasuk komponen non-tree yang ditambahkan oleh skema agregasi
federated.

Perbedaan admisibilitas tersebut tidak mengeluarkan ketiga model berbasis sampling
dari analisis konsistensi. Yang dituntutnya adalah pembandingan terhadap lantai
derau estimator, dan cara pembandingan itulah yang diuraikan pada sub-subbab
berikut.

=== Lantai derau per sel dan uji exchangeability eksak

Karena KernelSHAP menyampel koalisi secara acak, stabilitas antar client bagi FFD,
BERT, dan FedXGBllr hanya bermakna relatif terhadap kesepakatan KernelSHAP dengan
dirinya sendiri. Lantai derau tersebut diukur per client dan per sel, yaitu
kesepakatan antara dua vektor importance dari client yang sama pada dua seed
koalisi berbeda dengan explanation set dan background yang identik. Pengukuran
per sel diperlukan karena lantai bergantung pada jumlah fitur: pada
`nsamples` sebesar 500, PaySim mengenumerasi sekitar 54 persen bobot kernel, ULB
sekitar 26 persen, dan BAF sekitar 22 persen. Audit `analysis/verify_kernel_tier.py`
mencatat 56 nilai lantai yang berbeda pada grid ini, sehingga satu angka tunggal
tidak dapat mewakili keseluruhannya.

Lantai tersebut tidak dapat dipakai sebagai ambang deteksi, dan penegasan ini
mengoreksi pembacaan yang lazim. Lantai merupakan kesepakatan client yang sama pada
seed yang berbeda. Di bawah hipotesis nol bahwa seluruh client berbagi satu vektor
importance sejati, kesepakatan antar client yang diharapkan justru sama dengan
lantai, bukan berada di bawahnya. Nilai yang berada jauh di bawah lantai karenanya
merupakan bukti yang menentang hipotesis nol, sehingga rumusan "pada atau di bawah
lantai" menggabungkan dua keadaan yang menunjuk ke arah berlawanan dan tidak dapat
dipertahankan. Pengujian yang dipakai menggantikan pembacaan ambang tersebut dengan
uji exchangeability eksak: di bawah hipotesis nol, kesepuluh vektor pada satu sel
bersifat exchangeable, sehingga distribusi nolnya adalah seluruh perfect matching
atas kesepuluh vektor tersebut. Untuk $K = 5$ jumlahnya $(2K - 1)!! = 945$ dan
seluruhnya dienumerasi, bukan disampel dan bukan didekati secara asimptotik.
Nilai p terkecil yang dapat dicapai adalah 1/945 atau sekitar 0,00106, dan tidak
ada satu sel pun yang melanggar batas tersebut. Koreksi Benjamini--Hochberg
diterapkan per keluarga (bg, explainer).

Statistik utama yang dilaporkan adalah korelasi peringkat Spearman berbobot
magnitudo. Alasannya bersifat empiris. Pada seluruh 33 sel KernelSHAP tersampel,
kesepakatan dalam-client berbobot bernilai sekurang-kurangnya 0,9726, sedangkan
`floor_min` tanpa bobot berayun antara 0,8331 dan 0,9999. Ayunan tersebut bukan
cerminan ketidakstabilan estimator melainkan konsekuensi dari urutan yang sembarang
di antara fitur-fitur berkontribusi rendah yang nyaris berimbang; statistik tanpa
bobot didominasi oleh ekor tersebut, sedangkan statistik berbobot tidak. Ukuran
Jaccard\@5 dan indeks Kuncheva tetap dilaporkan sebagai pelengkap kontinuitas
terhadap pengukuran sebelumnya, dan @fig-4-shap-jaccard-kuncheva memperlihatkan
mengapa keduanya tidak dapat dipertukarkan: keduanya menyimpang seiring
bertambahnya dimensi, yang merupakan pembenaran empiris bagi koreksi peluang
menurut @nogueira2018stability.

#figure(
  image("resources/fig-shap-jaccard-vs-kuncheva.png", width: 62%),
  caption: [Jaccard\@5 terhadap indeks Kuncheva per sel, ditandai menurut dataset dengan dimensionalitasnya (PaySim $d = 13$, ULB $d = 30$, BAF $d = 55$). Kedua ukuran menyimpang seiring bertambahnya dimensi — pembenaran empiris untuk koreksi peluang @nogueira2018stability.],
) <fig-4-shap-jaccard-kuncheva>

Tiga penjaga melindungi pelaporan dari nilai kesepakatan yang bersifat artefak,
sebagaimana diuraikan pada Subbab Implementasi Modul Evaluasi dan SHAP: penjaga
degenerasi, penjaga keruntuhan sumbu client, dan penjaga regresi `l1_reg`. Audit
tingkat kernel mencatat bahwa tidak ada satu sel pun yang masih memperlihatkan
tanda tangan seleksi sepuluh fitur, dengan jumlah nonzero per baris minimum
sebesar 13 di seluruh grid.

=== Konsistensi feature importance antar client

Dengan lantai yang diukur per sel dan uji yang bersifat eksak, konsistensi feature
importance antar client dapat dijawab untuk keenam model, bukan hanya untuk ketiga
model dengan explainer deterministik. Sebanyak 29 dari 33 sel KernelSHAP tersampel
membawa nilai stabilitas yang dapat dibedakan secara statistik dari lantai derau
explainer-nya sendiri. Pada 19 sel di antaranya pemisahan bahkan bersifat lengkap,
yakni setiap nilai dalam-client berada di atas setiap nilai antar-client, sehingga
tidak memerlukan uji sama sekali untuk dibaca. @tab-4-shap-kernel merangkum ketiga
keluarga model tersebut.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([*Model*], [*Lantai (median)*], [*Antar-client (median)*],
                 [*Rentang*], [*Dapat dibedakan*]),
    [FFD], [0,9997], [0,963], [0,798 -- 0,992], [11 / 11],
    [BERT], [0,9993], [0,952], [0,868 -- 0,995], [10 / 11],
    [FedXGBllr], [0,9988], [0,958], [0,665 -- 0,986], [8 / 11],
  ),
  kind: table,
  caption: [Lantai derau dan stabilitas antar client pada tingkat KernelSHAP tersampel (Spearman berbobot magnitudo, 11 sel federated per model)],
) <tab-4-shap-kernel>

Gambaran yang muncul berbeda secara mendasar dari pengukuran sebelumnya. Ketiga
model berbasis sampling ternyata kira-kira sama terukurnya sekaligus kira-kira sama
stabilnya pada kisaran 0,95, dan tidak ada satu pun di antaranya yang menonjol
sebagai keluarga yang tidak stabil. Nilai FedXGBllr sebesar 0,775 yang dilaporkan
pada pengukuran sebelumnya merupakan artefak dari nilai bawaan `l1_reg` dan bukan
properti model. Sisi lain dari temuan ini adalah bahwa empat sel tetap tidak dapat
dibedakan dari derau estimator, dan keempatnya dilaporkan sebagai tidak konklusif,
bukan sebagai kesepakatan.

Nilai-nilai tersebut dapat dibandingkan terhadap sebuah jangkar, namun jangkar
itu harus dipilih dengan hati-hati. Satu-satunya jangkar yang sah adalah model
yang bersifat eksak secara independen dari model yang sedang diperiksa, yaitu LR
dan SVM dengan LinearSHAP serta GBM dengan TreeSHAP interventional; selisih
maksimum antar-seed pada seluruh 53 sel keluarga tersebut terukur tepat nol.
Tingkat eksak PaySim tidak dapat menjadi jangkar meskipun sama-sama bebas derau,
sebab isinya adalah ketiga model KernelSHAP yang sama yang dijalankan ulang tanpa
sampling; membandingkan tingkat tersampel terhadapnya berarti membandingkan
sebuah estimator terhadap versi eksak dirinya sendiri, dan perbandingan semacam
itu tidak mungkin gagal. Tingkat eksak karena itu dilaporkan sebagai tingkat
tersendiri pada sub-subbab Tingkat eksak pada PaySim, bukan sebagai pembanding.

Seluruh perbandingan berikut memakai statistik yang sama, yaitu Spearman berbobot
magnitudo. Pada jangkar, sebaran antar client bergerak antara 0,9064 dan 0,9970
untuk LR, antara 0,8980 dan 0,9989 untuk SVM, serta antara 0,8014 dan 0,9960 untuk
GBM, sehingga nilai paling divergen pada keseluruhan jangkar adalah 0,8014 yang
terjadi pada sel ULB GBM kondisi IID tanpa SMOTE. Dua dari 33 sel KernelSHAP
tersampel berada di bawah nilai tersebut, yaitu PaySim FedXGBllr Dirichlet dengan
SMOTE sebesar 0,6650 dan PaySim FFD Dirichlet tanpa SMOTE sebesar 0,7979.

Kedua sel tersebut merupakan temuan, bukan pemeriksaan validitas, dan pembedaan
itu perlu dinyatakan tegas. Keduanya bertahan pada tingkat eksak dengan nilai
0,5965 dan 0,7979, sehingga posisinya bukan artefak sampling. Yang terbaca adalah
bahwa pada partisi Dirichlet PaySim, kedua model berbasis sampling berselisih
antar client lebih jauh daripada sel mana pun pada keluarga LR, SVM, dan GBM.
Pengamatan tersebut terkonfound dengan keluarga model: model yang menerima
explainer eksak juga merupakan model yang paling sederhana secara struktural,
yakni koefisien linear dan prefix pohon dangkal, sehingga yang terukur adalah
keluarga model mana yang terpengaruh dan bukan mekanisme yang menyebabkannya.
@fig-4-rq3-stability menyajikan keseluruhan grid untuk keenam model.

#figure(
  image("resources/fig-4-rq3-stability.png", width: 100%),
  caption: [Stabilitas feature importance antar client untuk keenam model (Spearman berbobot magnitudo, $K = 5$, arm background per-client). Warna menyandikan divergensi. Sel berarsir tidak dapat dibedakan dari lantai derau explainer-nya menurut uji exchangeability eksak dan karenanya tidak membawa klaim stabilitas. Titik pada sudut sel menandai sel PaySim yang nilainya berasal dari tingkat eksak.],
) <fig-4-rq3-stability>

Pada tingkat deterministik, rincian per dataset mengungkap bahwa urutan antar model
bukan properti model semata. Rerata indeks Kuncheva pada sel federated adalah 0,9433
untuk SVM, 0,9112 untuk LR, dan 0,8219 untuk GBM dengan nilai minimum 0,544.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header([*Dataset*], [*GBM*], [*LR*], [*SVM*]),
    [BAF], [0,853], [0,956], [0,927],
    [ULB], [0,742], [1,000], [0,964],
    [PaySim], [0,878], [0,789], [0,935],
  ),
  kind: table,
  caption: [Indeks Kuncheva per dataset dan model pada tingkat deterministik],
) <tab-4-shap-kuncheva-det>

Pernyataan bahwa GBM merupakan model paling tidak stabil karena itu perlu
diperhalus. GBM paling tidak stabil pada ULB dengan 0,742, yaitu justru pada dataset
tempat LR mencapai stabilitas sempurna sebesar 1,000. Sebaliknya, LR paling tidak
stabil pada PaySim dengan 0,789, yaitu pada dataset tempat GBM justru relatif baik
dengan 0,878. Ketidakstabilan interpretasi dengan demikian merupakan interaksi antara
keluarga model dan distribusi data, bukan properti yang melekat pada salah satunya
saja.

Mekanisme yang menjelaskan interaksi tersebut bersifat konsisten dengan cara
masing-masing explainer memakai data background. Atribusi LinearSHAP bergantung pada
background hanya melalui rerata fitur, dan rerata tersebut serupa antar client ketika
distribusi fiturnya serupa. Fitur ULB berupa komponen PCA yang menurut konstruksinya
nyaris terdistribusi identik antar client, sehingga LR mencapai 1,000 pada dataset
tersebut. Struktur split pada model pohon berinteraksi dengan densitas lokal dengan
cara yang tidak dialami koefisien linear, sehingga GBM tetap sensitif terhadap
heterogenitas bahkan pada data yang membuat model linear sepenuhnya stabil.

=== Stabilitas interpretasi di bawah kondisi Non-IID

Pengaruh heterogenitas distribusi terhadap stabilitas interpretasi diukur pada dua
kanal yang terpisah, dan pemisahan itu diperlukan karena arm utama hanya
memvariasikan satu masukan. Arm utama menahan baris yang dijelaskan tetap, yakni
subset test terpusat yang identik bagi seluruh client, dan membiarkan background
bervariasi per client; kanal yang terisolasi di sana adalah distribusi referensi.
Arm background bersama melakukan kebalikannya: background disatukan menjadi satu
background terpool, sementara setiap client menjelaskan sampel dari partisi lokalnya
sendiri; kanal yang terisolasi di sana adalah wilayah data.

Uji yang dipakai pada kedua kanal identik dan dinyatakan lengkap di sini agar
hasilnya dapat direproduksi. Statistik yang diuji adalah Spearman berbobot
magnitudo per sel, yaitu kolom `weighted_between` pada
`results/shap_v2/shap_summary_v2.csv`, dan bukan selisihnya terhadap lantai derau;
mengurangkan lantai per sel justru menambahkan suku pengganggu karena lantai itu
sendiri berbeda antara arm Dirichlet dan arm IID. Ujinya adalah Wilcoxon
signed-rank satu sisi dengan hipotesis alternatif bahwa nilai IID lebih besar
daripada nilai Dirichlet. Pasangan dibentuk di dalam kombinasi (dataset, model,
arm) untuk kanal pertama sehingga n sama dengan 15, dan di dalam kombinasi
(model, arm) untuk kanal kedua yang hanya tersedia pada PaySim sehingga n sama
dengan 6.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header([*Kanal*], [*Yang bervariasi antar client*], [*Hasil*]),
    [1 — distribusi referensi],
    [background per client, baris yang dijelaskan tetap],
    [11 dari 15 pasangan; Wilcoxon p = 0,0042. Khusus PaySim 6 dari 6, p = 0,0156],
    [2 — wilayah data],
    [background terpool, baris per client],
    [PaySim 5 dari 6 pasangan; Wilcoxon p = 0,0469],
  ),
  kind: table,
  caption: [Dua kanal pengaruh heterogenitas terhadap divergensi interpretasi antar client (uji Wilcoxon signed-rank satu sisi berpasangan Dirichlet lawan IID)],
) <tab-4-shap-channels>

Selisih berpasangan rerata sebesar +0,054 pada kanal pertama dan +0,073 pada
kanal kedua. Kedua besaran tersebut *tidak dapat dibandingkan satu sama lain*, dan
hal ini perlu dinyatakan tegas agar tidak disalahtafsirkan sebagai perbandingan
kekuatan kanal. Kedua arm berbeda pada sumber background, pada sumber baris yang
dijelaskan, sekaligus pada apakah baris tersebut berasal dari data latih atau data
uji. Yang absah hanyalah kontras Dirichlet lawan IID *di dalam* masing-masing kanal.

Efek tersebut terkonsentrasi pada PaySim. Pada kanal pertama, keenam pasangan
PaySim seluruhnya berpihak pada Dirichlet, sedangkan ULB berpihak pada empat dari
enam pasangan dengan selisih yang kecil dan BAF hanya pada satu dari tiga pasangan
dengan satu pasangan yang praktis berimbang. Konsentrasi tersebut sebaiknya dibaca
sebagai gradien alih-alih sebagai saklar: PaySim pada $alpha$ sebesar 0,5 merupakan
partisi paling degeneratif dalam penelitian ini dengan jumlah minoritas per client
terendah sebesar 2, dibandingkan 3 pada BAF dan 4 pada ULB sebagaimana tercatat pada
`results/analysis/minority_census.csv`. Divergensi interpretasi dengan demikian
berskala mengikuti seberapa parah partisi mengeringkan client, dan pengamatan ini
menautkan sub-subbab ini pada temuan heterogenitas di Subbab Pengaruh Heterogenitas
Distribusi Data dan Penanganan Class Imbalance alih-alih membiarkannya berdiri
sendiri.

Bukti dari tingkat deterministik menunjuk ke arah yang sama. Nilai minimum indeks
Kuncheva pada seluruh sel deterministik sebesar 0,544 dan terjadi pada sel ULB GBM
dengan partisi Dirichlet, bukan pada sel IID mana pun. Satu pasangan yang dapat
dibandingkan secara langsung, yaitu PaySim GBM pada arm tanpa SMOTE, mencapai
stabilitas sempurna sebesar 1,000 pada kondisi IID namun turun menjadi 0,7725 pada
kondisi Dirichlet.

=== Tingkat eksak pada PaySim

Seluruh angka pada kedua kanal di atas berasal dari estimator berbasis sampling,
sehingga satu tingkat pengukuran tambahan dijalankan untuk memastikan bahwa
kesimpulannya bukan artefak estimator. Tingkat ini berdiri sendiri dan tidak
dipakai sebagai jangkar bagi tingkat tersampel, sebab isinya adalah ketiga model
yang sama; perannya adalah menghapus derau estimator dari kontras Dirichlet lawan
IID, bukan menyediakan pembanding yang independen. Dengan mengelompokkan kelima kolom one-hot
`type` menjadi satu pemain Shapley, PaySim memiliki $M$ efektif sama dengan 9,
sehingga `nsamples` sebesar $2^9 - 2 = 510$ mengenumerasi seluruh bobot kernel tanpa
satu pun undian acak. Keeksakan tersebut diverifikasi alih-alih diandaikan: pada
kedua belas sel federated tingkat ini, selisih maksimum absolut antara atribusi seed
11 dan seed 22 terukur tepat 0,000, sementara selisih antara client 0 dan client 1
bernilai tidak nol pada seluruh sel sehingga sumbu client tidak runtuh.

#figure(
  table(
    columns: 4,
    align: (left, left, center, center),
    table.header([*Model*], [*Arm*], [*IID*], [*Dirichlet*]),
    [FedXGBllr], [tanpa SMOTE], [1,000], [0,847],
    [FedXGBllr], [SMOTE], [0,961], [0,597],
    [FFD], [tanpa SMOTE], [0,874], [0,798],
    [FFD], [SMOTE], [0,912], [0,843],
    [BERT], [tanpa SMOTE], [0,998], [0,918],
    [BERT], [SMOTE], [0,970], [0,939],
  ),
  kind: table,
  caption: [Stabilitas antar client pada tingkat eksak PaySim (Spearman berbobot magnitudo, atribusi bebas derau estimator)],
) <tab-4-shap-exact>

Pada atribusi bebas derau tersebut, kondisi Dirichlet lebih rendah daripada kondisi
IID pada keenam pasangan, dengan Wilcoxon p = 1/64 = 0,0156 yang merupakan
nilai terkecil yang dapat dicapai pada $n = 6$.

Klaim tersebut selanjutnya dikorroborasi tanpa mengandalkan satu pun pilihan
metodologis yang dapat diperdebatkan. Ukuran berikut adalah selisih maksimum
absolut antara vektor atribusi client 0 dan client 1 pada ruang atribusi itu
sendiri: tanpa Spearman, tanpa lantai, tanpa uji permutasi, dan tanpa pembobotan.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header([*Model*], [*IID tanpa SMOTE*], [*Dirichlet tanpa SMOTE*],
                 [*IID SMOTE*], [*Dirichlet SMOTE*]),
    [BERT], [0,296], [1,740 (5,9 kali)], [0,623], [2,212 (3,6 kali)],
    [FFD], [0,475], [2,677 (5,6 kali)], [0,963], [2,543 (2,6 kali)],
    [FedXGBllr], [0,0045], [0,0878 (19,5 kali)], [0,0378], [0,100 (2,6 kali)],
  ),
  kind: table,
  caption: [Selisih maksimum absolut atribusi antara client 0 dan client 1 pada tingkat eksak PaySim, beserta rasio Dirichlet terhadap IID],
) <tab-4-shap-rawgap>

Keenam pasangan bergerak ke arah yang sama dengan rasio antara 2,6 dan 19,5 kali.
Karena setiap pilihan metodologis yang dapat diperdebatkan dilewati dan efeknya
tetap bertahan, tabel tersebut merupakan bukti terkuat pada bab ini.

Tingkat eksak sekaligus memungkinkan pengukuran langsung terhadap bias tingkat
tersampel. Pada kedua belas sel PaySim yang sama, selisih rerata absolut antara
nilai tersampel dan nilai eksak sebesar 0,028 dengan selisih maksimum 0,099 yang
terjadi pada sel FFD IID tanpa SMOTE, yang turun dari 0,973 menjadi 0,874. Tujuh
dari dua belas sel bergerak turun, satu sel berimbang tepat, dan empat sel bergerak
naik, sehingga tingkat tersampel secara neto melebih-lebihkan stabilitas dan
merendahkan divergensi; arah bias tersebut bersifat konservatif terhadap kesimpulan
yang dilaporkan.

Verdict statistik kedua tingkat sepakat pada sebelas dari dua belas sel.
Pengecualiannya adalah sel BERT IID tanpa SMOTE, yang menghasilkan p = 0,001 pada
tingkat tersampel namun p = 0,111 pada tingkat eksak. Sel tersebut merupakan
positif palsu pada tingkat tersampel, dan pelaporannya bersifat wajib karena
justru merupakan argumen terbaik mengenai alasan tingkat eksak dibangun. Untuk
PaySim, seluruh angka yang dikutip pada subbab ini berasal dari tingkat eksak;
angka tersampel dipertahankan sebagai pemeriksaan silang yang menetapkan besaran
biasnya.

=== Pengaruh SMOTE terhadap stabilitas interpretasi

Penanganan class imbalance memengaruhi stabilitas interpretasi dengan cara yang
memiliki dua sisi, dan kedua sisi itu perlu dibaca bersamaan agar tidak menyesatkan.

Sisi pertama adalah kesepakatan antar client yang meningkat. Pada model
deterministik, rerata indeks Kuncheva naik dari 0,8657 pada arm tanpa SMOTE menjadi
0,9238 pada arm dengan SMOTE, sedangkan nilai minimumnya naik dari 0,544 menjadi
0,7725 (@tab-4-shap-arm). Kenaikan tersebut konsisten pada enam dari sembilan
pasangan dataset dan model, dengan dua pasangan yang datar karena telah jenuh, yaitu
ULB LR pada 1,000 dan PaySim SVM pada 0,935. Pasangan kesembilan berupa PaySim GBM
sedikit menurun sebesar 0,016, namun penurunan itu bukan anomali: sel IID-nya turun
dari 1,000 menjadi 0,870 sementara sel Dirichlet-nya naik dari 0,7725 menjadi 0,870,
sehingga keduanya bergerak menuju nilai tengah yang sama dari arah yang berlawanan.
Pergerakan tersebut persis yang diprediksi oleh mekanisme homogenisasi, yakni bahwa
SMOTE menarik stabilitas menuju sebuah nilai tengah terlepas dari titik awalnya, dan
sebuah pembalikan yang sesuai dengan mekanisme merupakan bukti yang lebih kuat
daripada kenaikan yang seragam.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    table.header([*Agregat (deterministik)*], [*Tanpa-SMOTE*], [*Dengan-SMOTE*]),
    [Rerata Kuncheva], [0,8657], [0,9238],
    [Minimum Kuncheva], [0,544], [0,7725],
    [Kenaikan rerata — IID], table.cell(colspan: 2, align: center)[$+0,048$],
    [Kenaikan rerata — Dirichlet], table.cell(colspan: 2, align: center)[$+0,085$],
  ),
  kind: table,
  caption: [Indeks Kuncheva menurut arm SMOTE],
) <tab-4-shap-arm>

Sisi kedua adalah bahwa dasar kesepakatan itu berubah. Fitur paling penting menurut
`top_feature` berbeda antar arm pada lima dari 15 pasangan sel yang sebanding, dan
ketika dibatasi pada partisi Dirichlet perbedaan itu terjadi pada tiga dari sembilan
pasangan. Konsentrasi perubahan di bawah Dirichlet mendukung mekanisme yang sama.
Yang terjadi karena itu bukan sekadar kesepakatan yang lebih kuat atas fitur yang
sama, melainkan kesepakatan atas himpunan fitur yang berbeda. @tab-4-shap-shift
menyajikan tiga kasus yang telah diverifikasi.

#figure(
  text(size: 8pt)[
    #set par(justify: false)
    #table(
      columns: (auto, 1fr, 1fr),
      align: left + top,
      table.header([*Sel*], [*Top-5 tanpa SMOTE*], [*Top-5 dengan SMOTE*]),
      [ULB GBM],
      [Seluruh client membuka `V14` lalu `V7`, kemudian menyebar ke `V4`, `V15`,
      `V19`, `V12`, `V10`, `V26`, dan `V17`],
      [Kelima client membuka `V4`, `V14`, lalu `V3`; empat di antaranya berbagi
      `V24` atau `V28` pada peringkat keempat dan kelima],

      [BAF LR],
      [`prev_address_months_count_missing` menempati peringkat teratas pada seluruh
      client],
      [`housing_status_BB` menggeser indikator missing dari peringkat teratas],

      [BAF SVM],
      [Campuran `prev_address_months_count_missing`, `housing_status_BA`,
      `housing_status_BE`, `has_other_cards`, `employment_status_CA`, dan
      `device_os_windows`],
      [Empat dari lima teratas berupa kolom `housing_status_*`, yaitu `BB`, `BA`,
      `BC`, dan `BE`, ditambah `has_other_cards`],
    )
  ],
  kind: table,
  caption: [Pergeseran fitur top-5 antar arm SMOTE pada tiga sel Dirichlet],
) <tab-4-shap-shift>

Pergeseran tersebut dapat ditelusuri pada tingkat fitur individual. Pada ULB GBM,
yang indeks Kunchevanya naik dari 0,544 menjadi 0,904 di bawah SMOTE, fitur V7 yang
pada arm tanpa SMOTE menempati peringkat kedua secara bulat menghilang dari setiap
daftar, sementara V3 yang semula absen dari seluruh daftar justru menjadi peringkat
ketiga secara universal. Kasus paling tajam terjadi pada BAF SVM, tempat
empat dari lima fitur teratas di bawah SMOTE merupakan kolom one-hot dari kelompok
`housing_status`. Keterbatasan one-hot yang telah didokumentasikan pada Subbab
Perancangan Skema Class Imbalance Handling dan pada pembahasan SMOTE di dasar teori
memperlihatkan konsekuensi teramatinya di sini: SMOTE standar menginterpolasi secara
kontinu melintasi kolom one-hot yang saling eksklusif sehingga menghasilkan rekaman
sintetis dengan nilai pecahan pada beberapa indikator sekaligus, yakni sebuah
transaksi yang sebagiannya menyandang beberapa status hunian pada saat bersamaan.
Model menemukan gradien buatan tersebut informatif, dan atribusi kemudian
terkonsentrasi pada blok kolom itu. SMOTE-NC menurut
#cite(<chawla2002smote>, form: "prose") merupakan remedi yang dimaksudkan untuk kasus
semacam ini, dan pengamatan ini menaikkan rekomendasi tersebut dari sekadar sitasi
menjadi temuan empiris.

Satu klaim perlu dinyatakan secara hati-hati agar tidak disalahtafsirkan. Kesepakatan
interpretasi antar client bukan bukti bahwa model federated telah mempelajari struktur
bersama, sebab oversampling lokal terbukti dapat memanufaktur kesepakatan sekaligus
mengubah apa yang disepakati. Digabungkan dengan analisis performa, tempat intervensi
yang sama membuat kedua model deep kehilangan antara 71 dan 73 persen AUPRC, gambaran
yang muncul adalah model yang menjadi lebih konsisten dan lebih tampak percaya diri
sekaligus lebih buruk. Pengamatan ini diukur pada sembilan pasangan dataset dan model
dengan satu seed serta tiga contoh terperinci, sehingga statusnya adalah observasi
terdokumentasi dengan mekanisme yang konsisten, bukan hukum umum yang telah terbukti.

=== Degenerasi dan kasus batas

Dua sel PaySim FedXGBllr pada partisi Dirichlet mula-mula menghasilkan atribusi
bernilai nol untuk seluruh fitur pada setiap client. Akar penyebabnya adalah wrapper
explanation yang menghitung probabilitas lalu menerapkan transformasi logit dengan
klip pada 1e−6, sedangkan probabilitas PaySim FedXGBllr berada di sekitar 1e−9 yang
berada di bawah batas klip tersebut, sehingga setiap prediksi jenuh pada satu
konstanta dan setiap perturbasi fitur tidak menggeser luaran. Persoalan ini
diperbaiki dengan mengekspos aktivasi pra-Sigmoid secara langsung, dan setelah
perbaikan kedua sel tersebut merupakan sel paling divergen dalam keseluruhan studi,
dengan Spearman berbobot sebesar 0,822 pada arm tanpa SMOTE dan 0,665 pada arm
dengan SMOTE. Degenerasi ini merupakan kompresi
probabilitas yang sama yang membuat kalibrasi menjadi tidak terdefinisi pada Subbab
Perbandingan Diskriminasi dan Kalibrasi, sehingga satu degenerasi yang sama
menampakkan diri di dua tempat dan keduanya teratasi dengan bekerja pada skala
logit.

Penjaga degenerasi ditambahkan sebagai konsekuensinya. Vektor client yang seluruhnya
bernilai nol atau konstan kini menghasilkan status tidak terdefinisi alih-alih
menghasilkan metrik. Sebelum penjaga tersebut dipasang, vektor nol menghasilkan
Jaccard 1,0 dan Kuncheva 1,0, yakni kesepakatan sempurna yang semu atas ketiadaan.
Status tidak terdefinisi dan nilai 0,0 karena itu tidak boleh disamakan, sebab 0,0
berarti seluruh client sepenuhnya tidak sepakat sedangkan status tidak terdefinisi
berarti peringkatnya memang degeneratif sehingga korelasi tidak dapat dihitung.

Dua kasus batas lain perlu dicatat agar pembacaan agregat tidak keliru. Sel GBM
dengan prefix satu pohon berkedalaman enam memiliki atribusi yang terbatas secara
struktural, dan keterbatasan itu mengikuti seleksi iterasi pada validation set
sebagaimana diuraikan pada Subbab Implementasi Pelatihan Model dan Skema Agregasi
alih-alih merupakan sebuah kesalahan. Sel centralized hanya memiliki satu client
sehingga tidak memiliki stabilitas antar client menurut definisi dan tidak masuk ke
dalam agregat mana pun.

=== Keterbatasan pengukuran interpretabilitas

Tujuh keterbatasan membatasi jangkauan klaim pada subbab ini dan dinyatakan secara
eksplisit agar tidak terbaca lebih luas daripada yang diukur.

Pertama, seluruh pengukuran bersandar pada satu seed pelatihan, yaitu 42. Divergensi
diukur pada satu model terlatih, bukan lintas undian model. Keterbatasan ini telah
ada sebelumnya dan turut membatasi klaim ini.

Kedua, empat sel KernelSHAP tersampel bersifat tidak konklusif alih-alih menunjukkan
kesepakatan, yaitu ULB FedXGBllr pada kondisi IID dan Dirichlet tanpa SMOTE, PaySim
FedXGBllr pada kondisi IID tanpa SMOTE, serta ULB BERT pada kondisi IID tanpa SMOTE.
Tiga di antaranya berasal dari keluarga FedXGBllr yang sama, namun beberapa upaya
menjelaskan keluarga tersebut melalui satu statistik ringkas — fraksi atribusi
mendekati nol, entropi profil, maupun median magnitudo atribusi — seluruhnya gagal.
Pola tersebut dilaporkan sebagai teramati, bukan sebagai terjelaskan.

Ketiga, kanal kedua hanya tersedia pada PaySim. ULB dan BAF tidak memiliki arm
background bersama.

Keempat, kedua arm bukan rancangan satu faktor. Sumber background, sumber baris yang
dijelaskan, dan asal baris tersebut dari data latih atau data uji seluruhnya berbeda
antar arm, sehingga besaran kedua kanal tidak dapat dibandingkan. Rancangan satu
faktor yang bersih menuntut arm ketiga yang memvariasikan baris per client sekaligus
background per client, yakni kondisi yang sesungguhnya terjadi pada penerapan nyata;
arm tersebut tidak dijalankan. Karena kedua kanal bertanda positif, kasus gabungannya
sekurang-kurangnya sama besar, dan simpulan tersebut dinyatakan sebagai inferensi
alih-alih sebagai hasil pengukuran.

Kelima, arm background bersama menjelaskan baris data latih. Partisi client hanya
mencakup `x_train` dan tidak tersedia split tertahan per client pada pipeline ini,
sehingga SHAP menjelaskan fungsi yang telah dilatih dan bukan kemampuan
generalisasinya.

Keenam, tingkat eksak hanya tersedia pada PaySim dan hanya dalam bentuk terkelompok.
ULB menuntut $2^30$ koalisi dan BAF menuntut $2^55$. Pengelompokan yang sama pada BAF
tetap menghasilkan $M$ efektif antara 29 dan 34, sehingga pengelompokan di sana tidak
membeli keeksakan melainkan hanya argumen kebenaran yang berdekatan dengan SMOTE-NC
serta reduksi variansi.

Ketujuh, rumusan masalah ketiga menanyakan konsistensi peringkat, sedangkan peringkat
dan magnitudo tidak selalu bergerak bersama. Sel PaySim FedXGBllr pada kondisi IID
tanpa SMOTE merupakan tempat perbedaan itu paling terasa: kedua client berbeda pada
magnitudo atribusi sebesar 4,5e−3 yang bernilai sekitar sembilan kali median
atribusi sel tersebut, sementara urutannya nyaris tidak berbeda dengan Spearman
berbobot sebesar 0,999998.

=== Sintesis temuan interpretabilitas antar keluarga model

Karakteristik explainability ketiga keluarga model berbeda pada dua tingkat yang
perlu dipisahkan, yaitu tingkat admisibilitas dan tingkat konsistensi. Pada tingkat
admisibilitas, model parametrik dan model pohon murni menerima explainer eksak dengan
galat local accuracy sebesar 9,26e−8 dan nol, sedangkan model deep tidak menerima
estimator eksak apa pun karena DeepSHAP tidak memiliki aturan propagasi untuk
LayerNorm dan GradientExplainer meleset sekitar 1.360 kali di atas toleransi. Temuan
yang paling menonjol adalah bahwa FedXGBllr, meskipun berbasis tree,
tidak mewarisi keeksakan TreeSHAP karena kepala agregasi CNN-nya memuat ReLU yang
memutus linearitas Shapley, sehingga model tersebut harus dijelaskan secara
model-agnostik sejajar dengan model deep. Kategori berbasis tree karena itu tidak
menentukan karakteristik explainability secara otomatis; yang menentukan adalah
keseluruhan jalur komputasi termasuk komponen non-tree yang ditambahkan oleh skema
agregasi federated.

Pada tingkat konsistensi, perbedaan admisibilitas tersebut ternyata tidak
menghalangi pengukuran. Dengan lantai derau yang diukur per client dan per sel serta
uji exchangeability yang bersifat eksak, konsistensi antar client terjawab untuk
keenam model: 29 dari 33 sel KernelSHAP tersampel membawa nilai stabilitas yang
dapat dibedakan dari lantai derau explainer-nya sendiri, dan ketiga model berbasis
sampling menempati kisaran stabilitas yang kira-kira sama pada 0,95. Pernyataan
sebelumnya bahwa konsistensi hanya terukur pada explainer deterministik merupakan
konsekuensi dari lantai tunggal yang disiarkan ke seluruh dataset dan dari nilai
bawaan `l1_reg` yang menihilkan sebagian besar vektor atribusi, bukan konsekuensi
dari sifat estimator berbasis sampling.

Perbedaan konsistensi antar keluarga model bersifat interaksi dengan distribusi data
alih-alih properti mutlak. Model pohon paling tidak stabil pada ULB dengan 0,742
justru di tempat model linear mencapai 1,000, sedangkan model linear paling tidak
stabil pada PaySim dengan 0,789 di tempat model pohon relatif baik dengan 0,878.
Mekanismenya terletak pada cara masing-masing explainer memakai data background:
LinearSHAP hanya menyerap background melalui rerata fitur yang serupa antar client
ketika distribusi fiturnya serupa, sedangkan struktur split pohon berinteraksi dengan
densitas lokal secara langsung.

Di bawah kondisi Non-IID, heterogenitas distribusi menaikkan divergensi interpretasi
antar client pada kedua kanal yang diukur secara terpisah, dan kesimpulan itu
bertahan pada atribusi yang sama sekali bebas derau estimator: pada tingkat eksak
PaySim, keenam pasangan Dirichlet berada di bawah pasangan IID-nya, dan selisih
atribusi mentah antar client membesar antara 2,6 dan 19,5 kali. Efeknya berskala
mengikuti keparahan partisi alih-alih bersifat seragam, sehingga temuan ini terhubung
langsung dengan temuan heterogenitas pada subbab sebelumnya.

Simpulan terakhir menyangkut hubungan antara stabilitas interpretasi dan robustnes
performa, yang ternyata tidak searah. Paradigma best-model selection yang mendasari
GBM merupakan paradigma paling tahan terhadap SMOTE dalam dimensi performa dengan
kehilangan AUPRC nol persen, namun justru paling tidak stabil dalam dimensi
interpretasi. Sebaliknya, penerapan SMOTE menaikkan rerata indeks Kuncheva dari 0,8657
menjadi 0,9238 sekaligus mengubah fitur paling penting pada lima dari 15 pasangan sel,
sehingga menghasilkan model yang lebih konsisten penjelasannya namun menjelaskan
sesuatu yang berbeda dan berperforma lebih buruk. Kesepakatan antar client karena itu
tidak dapat diperlakukan sebagai bukti bahwa model federated telah mempelajari
struktur bersama, dan evaluasi Explainable Federated Learning menuntut dimensi
performa serta dimensi interpretasi dilaporkan berdampingan alih-alih salah satunya
dipakai sebagai proksi bagi yang lain.

// ---------------------------------------------------------------------------
// BAB 5 — PENUTUP  (stub)
// ---------------------------------------------------------------------------

= PENUTUP

// TODO: BAB 5 belum ada di Proposal TA v2.1. Isi dengan Kesimpulan (menjawab
// ketiga rumusan masalah) dan Saran untuk penelitian lanjutan.

// ===========================================================================
// DRAF SARAN — disisipkan atas instruksi eksplisit penulis. Kesimpulan (yang
// menjawab rumusan masalah) BELUM ditulis karena Bab 4 belum ada; blok ini
// TIDAK memuat hasil, angka, atau temuan eksperimen. Tingkat keyakinan tiap
// butir sengaja dibedakan — jangan diratakan. Tinjau ulang setelah Bab 4.
// ===========================================================================

== Saran (draf — menunggu hasil Bab 4)

Blok berikut merupakan draf arah penelitian lanjutan pada tingkat keyakinan yang
berbeda-beda, disusun sebelum hasil eksperimen tersedia dan perlu ditinjau ulang
setelah Bab 4 rampung.

+ *Gerbang oversampling berdasarkan jumlah minoritas nyata minimum per client.*
  Mengikuti @blagus2013smote dan @elreedy2019smote yang menunjukkan degradasi
  SMOTE pada jumlah minoritas kecil, namun ini merupakan inferensi penelitian ini
  sendiri: belum ada penelitian terdahulu yang mengusulkan gerbang semacam itu
  untuk FL, dan tidak ada ambang numerik yang dapat disitasi.

+ *Random undersampling sebagai alternatif.* Paling didukung untuk regime ini —
  @blagus2013smote membandingkannya secara langsung dan mengunggulkannya pada data
  berdimensi tinggi. Teknik ini dikecualikan dari penelitian ini sesuai Batasan
  Masalah.

+ *Pelatihan lokal berbasis biaya (cost-sensitive).* Masuk akal namun masih
  diperdebatkan. Literatur FL cenderung menempuh penanganan pada level fungsi loss
  (@wang2021fedimbalance, @duan2019astraea), tetapi @weiss2007costsensitive tidak
  menemukan pemenang yang konsisten antara cost-sensitive learning dan sampling
  (pada venue minor). Arah ini terbuka dan belum tuntas.

+ *Migrasi ke SMOTE-NC* untuk data bertipe campuran, sebagaimana diperkenalkan
  #cite(<chawla2002smote>, form: "prose"), guna menghindari nilai pecahan pada
  kolom kategorikal hasil one-hot. Rekomendasi ini kini memiliki dukungan empiris,
  bukan sekadar sitasi: pada @sec-hasil-rq3 di bawah SMOTE atribusi BAF SVM
  terkonsentrasi pada empat kolom `housing_status_*` sekaligus, konsekuensi teramati
  dari interpolasi kontinu SMOTE standar melintasi blok one-hot yang saling eksklusif.

+ *Jangan memakai kesepakatan SHAP antar-client sebagai bukti struktur yang
  dipelajari* tanpa mengontrol oversampling. Pada @sec-hasil-rq3 SMOTE lokal
  menaikkan kesepakatan interpretasi antar-client sekaligus menggeser fitur yang
  disepakati, sehingga kenaikan kesepakatan dapat termanufaktur alih-alih
  mencerminkan struktur bersama yang benar-benar dipelajari model federated.

+ *Melaporkan kalibrasi berdampingan dengan diskriminasi* pada penelitian
  imbalance FL selanjutnya, mengikuti @goorbergh2022harm.

Varian SMOTE yang membatasi seed pada perbatasan (@han2005borderline,
@bunkhumpornpat2009safelevel) serta hibrida pembersihan (@batista2004balancing)
diperkirakan tidak membantu pada kondisi sekitar 21 seed, karena mayoritas seed
akan tergolong sebagai *danger* atau *noise* — hal ini merupakan konsekuensi dari
definisi algoritma-algoritma tersebut, bukan hasil yang telah diuji secara
empiris.

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
