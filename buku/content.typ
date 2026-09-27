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
#set math.equation(supplement: [Persamaan], numbering: n => {
  let ch = counter(heading.where(level: 1)).get().first()
  numbering("(1.1)", ch, n)
})

// ---------------------------------------------------------------------------
// BAB 1 — PENDAHULUAN
// ---------------------------------------------------------------------------

= PENDAHULUAN

== Latar Belakang

Perkembangan transformasi digital sektor keuangan telah mendorong peningkatan
volume transaksi elektronik sekaligus memperluas ancaman kecurangan finansial
(financial fraud). Laporan Association of Certified Fraud Examiners (ACFE)
menunjukkan bahwa organisasi global kehilangan rata-rata 5% dari pendapatan
tahunan mereka akibat fraud, dengan kerugian median per kasus mencapai
USD 145.000 @acfe2024. Di sisi lain, kompleksitas modus financial fraud terus
berkembang sehingga pendekatan deteksi tradisional berbasis rule-based menjadi
semakin tidak memadai karena keterbatasannya dalam menangkap pola anomali yang
dinamis dan terus beradaptasi @hilal2022. Maka dari itu, pendekatan berbasis
machine learning (ML) telah digunakan secara luas untuk mendeteksi pola fraud
melalui analisis data historis transaksi.

Adapun sebagian besar implementasi ML pada fraud detection masih bersifat
terpusat (centralized) yang mengharuskan agregasi data mentah dari berbagai
sumber institusi ke satu titik terpusat. Implementasi ini menimbulkan masalah
terkait privasi, kerahasiaan data nasabah, dan kepatuhan terhadap regulasi
seperti General Data Protection Regulation (GDPR) serta Undang-Undang Nomor 27
Tahun 2022 tentang Perlindungan Data Pribadi (PDP). Hal ini menghambat
kolaborasi antar lembaga keuangan untuk menghasilkan sumber data yang global
untuk membangun model deteksi fraud yang robust.

Federated Learning (FL) muncul sebagai pendekatan alternatif yang memungkinkan
beberapa pihak melatih model secara kolaboratif tanpa memindahkan data mentah
dari masing-masing sumber @mcmahan2023fedavg. Dalam pendekatan ini, pelatihan
model dilakukan secara lokal pada setiap client dan hanya parameter atau
pembaruan model yang dipindahkan ke server pusat untuk diagregasi. Pendekatan
ini relevan dalam sektor keuangan karena memungkinkan kolaborasi antar institusi
dengan menghindari perpindahan data mentah selama pelatihan meskipun perpindahan
parameter atau model tetap memerlukan perhatian terhadap kebocoran informasi
@kairouz2021. Beberapa studi terkini telah mengeksplorasi penerapan FL dalam
fraud detection, seperti pendekatan berbasis Graph Neural Network @tang2024
serta integrasi Explanaible Artificial Intelligence (XAI) untuk meningkatkan
transparansi model @aljunaid2025.

Walaupun demikian, implementasi FL untuk sektor keuangan masih menghadapi tiga
tantangan fundamental. Pertama, distribusi data antar institusi umumnya bersifat
non-independent and identically distributed (Non-IID) karena setiap bank
memiliki karakteristik nasabah, segmentasi pasar, dan profil risiko yang
berbeda. Heterogenitas data ini terbukti menurunkan performa model FL terutama
pada algoritma agregasi standar seperti FedAvg @li2021noniid. Kedua, fraud
detection juga menghadirkan permasalahan class imbalance yang ektrem dimana
proporsi transaksi fraud terhadap transaksi normal seringkali kurang dari 1%
@lopezrojas2016paysim. Hal ini menjadi lebih parah dalam implementasi FL karena
beberapa client mungkin memilki sangat sedikit atau bahkan tidak memiliki sampel
kelas minoritas. Ketiga, sebagian besar mekanisme agregasi FL yang sudah ada
dirancang khusus untuk model berbasis gradient descent, sementara model berbasis
tree yang justru terbukti unggul untuk data tabular masih kurang terintegrasi
dalam implementasi FL @grinsztajn2022.

Model berbasis tree, khususnya Extreme Gradient Boosting (XGBoost), telah
menjadi pilihan utama untuk klasifikasi pada data tabular karena kemampuannya
menangani missing values, fitur heterogen, dan interaksi non-linear secara
efisien @chen2016xgboost. Karakteristik yang dimiliki model berbasis tree ini
sangat relevan untuk fraud detection yang umumnya berbasis data transaksional
terstruktur. Namun, implementasi XGBoost dalam FL mengalami masalah yang
mendasar dimana pelatihan XGBoost konvensional dalam FL mengharuskan pertukaran
gradient dan hessian per-node. Hal ini mengakibatkan risiko kebocoran privasi
karena gradient dapat digunakan untuk merekonstruksi data mentah dan juga
mengakibatkan frekuensi komunikasi yang sangat tinggi bergantung pada kedalaman
dan jumlah @zhu2019deepleakage. Sebagai solusi,
#cite(<ma2023fedxgbllr>, form: "prose") memperkenalkan FedXGBllr (Federated
XGBoost with Learnable Learning Rates), yaitu sebuah kerangka gradient-less yang
mengagregasi tree ensembles antar client dan mempelajari bobot kontribusi setiap
pohon melalui one-layer 1D Convolutional Neural Network. Pendekatan ini terbukti
menurunkan communication overhead hingga 25-700 kali lipat sekaligus
menghilangkan kebutuhan untuk berbagi gradient.

Meskipun FedXGBllr menawarkan kontribusi yang signifikan sebagai solusi dalam
implementasi FL untuk fraud detection, evaluasi original FedXGBllr dilakukan
pada dataset publik berskala umum seperti HIGGS, SUSY, dan a9a yang tidak
memiliki karakteristik sektor keuangan. Dataset-dataset tersebut tidak memiliki
class imbalance ekstrem maupun struktur fitur transaksional pada fraud detection
sehingga performa FedXGBllr pada kondisi partisi Non-IID dengan rasio kelas
minoritas di bawah 1% masih belum dieksplorasi. Selain itu, studi lainnya
seperti #cite(<aljunaid2025>, form: "prose") yang telah menerapkan FL untuk
fraud detection dengan model konvensional seperti Logistic Regression, Support
Vector Machine, dan Gradient Boosting Machine dengan agregasi best-model
selection belum membandingkan dengan kerangka tree-based modern seperti
FedXGBllr.

Selain model parametrik dan model berbasis tree, implementasi model deep
learning pada FL juga dipertimbangkan untuk data tabular. Model FFD menggunakan
arsitektur 1D Convolutional Neural Network sedangkan model tabular Transformer
berbasis FT-Transformer yang disebut dengan BERT @gorishniy2021fttransformer.
Penambahan kedua model tersebut memperluas perbandingan agar mencakup
karakteristik model linear, model berbasis tree, dan model berbasis deep
learning dalam menghadapi distribusi data yang heterogen dalam konteks financial
fraud.

Dalam sektor keuangan, aspek lain yang juga menjadi perhatian adalah
explainability. Model fraud detection tidak cukup hanya akurat tetapi juga harus
dapat dijelaskan kepada auditor, regulator, dan juga nasabah @bussmann2021.
Shapley Additive exPlanations (SHAP) telah menjadi standar interpretasi model ML
karena dasar game theory yang konsisten @lundberg2017shap. Namun, penerapan SHAP
pada lingkungan FL menimbulkan pertanyaan terkait stabilitas hasil interpretasi
antar client ketika model dilatih di bawah kondisi Non-IID dan pengaruh
distribusi data antar client pada feature importance yang diperoleh.
#cite(<aljunaid2025>, form: "prose") telah mempertimbangkan XAI pada FL namun
belum menganalisis secara spesifik pengaruh mekanisme agregasi yang berbeda
memengaruhi stabilitas interpretasi antar client di bawah heterogenitas data.

Berdasarkan penjelasan tersebut, terdapat tiga celah penelitian yang saling
terkait. Pertama, minimnya eksplorasi model berbasis tree dalam FL untuk
financial fraud detection khususnya pada kerangka gradient-less seperti
FedXGBllr. Kedua, belum adanya perbandingan sistematis antara model dengan
agregasi gradient (LR dan SVM dengan FedAvg), model deep learning dengan
agregasi accuracy-weighted FedAvg, dan model dengan agregasi tree ensemble
(FedXGBllr) dalam satu kerangka eksperimen yang terkontrol. Ketiga, belum adanya
analisis mengenai dampak FL pada kondisi Non-IID dan class imbalance ekstrem
terhadap stabilitas interpretasi SHAP antar client.

Penelitian ini dilakukan untuk menjawab ketiga celah tersebut melalui evaluasi
komparatif yang sistematis. Dataset PaySim @lopezrojas2016paysim, yang merupakan
simulasi transaksi mobile money dengan rasio fraud sekitar 0,13%, digunakan
sebagai benchmark karena karakteristiknya yang merepresentasikan tantangan nyata
deteksi fraud pada sektor keuangan. Selain itu, digunakan pula dataset ULB
Credit Card yang bersifat riil namun dengan fitur yang dianonimkan, serta
dataset Bank Account Fraud (BAF) @jesus2022baf, yaitu data sintetis yang
dikembangkan berdasarkan data nyata dengan fitur bermakna sehingga memperkuat
interpretabilitas. Skenario Non-IID dibentuk melalui partisi Dirichlet untuk
merefleksikan heterogenitas data antar instirusi dan kerangka Flower
@beutel2022flower digunakan sebagai infrastruktur simulasi FL. Dengan demikian,
hasil penelitian ini diharapkan tidak hanya memperkaya literatur FL berbasis
tree tetapi juga memberikan landasan praktis bagi industri perbankan, otoritas
regulator seperti Otoritas Jasa Keuangan (OJK) dan Pusat Pelaporan dan Analisis
Transaksi Keuangan (PPATK), serta pengembang sistem deteksi fraud yang harus
menyeimbangkan akurasi, privasi, dan transparansi model.

== Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, penelitian ini merumuskan tiga
pertanyaan penelitian sebagai berikut:

+ Bagaimana perbandingan performa model Federated Learning berbasis tree
  (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support
  Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine dengan
  agregasi best-model selection, serta model deep learning (FFD berbasis 1D-CNN
  dan BERT berbasis tabular Transformer) dengan agregasi accuracy-weighted
  FedAvg dalam mendeteksi fraud finansial?
+ Bagaimana pengaruh heterogenitas distribusi data (IID dan Non-IID) serta
  penanganan class imbalance (SMOTE) terhadap performa model dalam federated
  learning?
+ Bagaimana karakteristik explainability model FL berbasis tree, model berbasis
  gradient, dan model deep learning, ditinjau dari konsistensi feature
  importance berbasis SHAP antar client dan stabilitasnya di bawah kondisi
  Non-IID?

== Batasan Masalah

Agar penelitian ini terfokus, terarah, dan dapat dipertanggungjawabkan secara
metodologis, ditetapkan sejumlah batasan masalah sebagai berikut:

+ *Batasan Domain dan Dataset.* Penelitian ini dibatasi pada deteksi financial
  fraud menggunakan tiga dataset publik, yaitu Financial Fraud Detection Dataset
  yang merupakan turunan simulator PaySim @lopezrojas2016paysim, ULB Credit Card
  Fraud Detection Dataset, dan Bank Account Fraud (BAF) varian Base
  @jesus2022baf. Domain fraud lain seperti insurance fraud dan anti-money
  laundering tidak dievaluasi. Perbandingan antar model dilakukan pada pembagian
  data evaluasi yang sama dalam masing-masing dataset.
+ *Batasan Model yang Dievaluasi.* Model yang dievaluasi dalam penelitian ini
  terbatas pada enam algoritma, yaitu Logistic Regression (LR) dan Support
  Vector Machine (SVM) sebagai representasi model parametrik dengan agregasi
  FedAvg, Gradient Boosting Machine (GBM) sebagai representasi model tree-based
  dengan agregasi best-model selection mengikuti skema
  #cite(<aljunaid2025>, form: "prose"), Financial Fraud Detection network (FFD)
  berupa 1D Convolutional Neural Network dan sebuah tabular Transformer
  (FT-Transformer, dilabeli BERT) sebagai representasi model deep learning,
  serta FedXGBllr @ma2023fedxgbllr sebagai representasi model tree-based dengan
  agregasi tree ensemble berbasis learnable learning rates. Pada kondisi
  terpusat, XGBoost digunakan sebagai pembanding bagi FedXGBllr, tanpa
  menyamakannya dengan implementasi terpusat dari model gabungan FedXGBllr.
  Model lain seperti Random Forest atau pendekatan berbasis Graph Neural Network
  (GNN) tidak dievaluasi karena di luar fokus perbandingan paradigma agregasi
  yang menjadi inti penelitian ini.
+ *Batasan Skema Agregasi Federated Learning.* Skema agregasi yang digunakan
  dalam penelitian ini terbatas pada empat mekanisme, yaitu FedAvg untuk model
  parametrik (LR, SVM), best-model selection untuk GBM, accuracy-weighted FedAvg
  untuk model deep learning (FFD, BERT), serta tree ensemble aggregation dengan
  learnable learning rates untuk FedXGBllr. Pada accuracy-weighted FedAvg,
  metrik kinerja lokal yang digunakan untuk pembobotan adalah AUPRC. Skema
  agregasi lanjutan seperti FedProx, SCAFFOLD, FedAvgM, atau FedNova tidak
  dievaluasi karena di luar ruang lingkup perbandingan dan akan memperluas
  dimensi eksperimen melebihi kapasitas penelitian ini.
+ *Batasan Skenario Heterogenitas Data.* Heterogenitas data antar client
  disimulasikan melalui partisi Dirichlet sebagai representasi label
  distribution skew. Eksperimen pelatihan utama membandingkan kondisi IID dengan
  Non-IID pada $alpha$ = 0,5. Variasi $alpha$ pada sensus diagnostik partisi
  tidak diperlakukan sebagai variasi eksperimen pelatihan utama. Feature
  distribution skew, concept drift, dan quantity skew murni tidak dievaluasi
  sebagai faktor eksperimen tersendiri.
+ *Batasan Lingkungan Eksperimen.* Penelitian ini menggunakan simulasi Federated
  Learning berbasis Flower @beutel2022flower dengan lima client dalam konteks
  cross-silo FL. Implementasi FedXGBllr mengikuti baseline hfedxgboost pada
  repositori resmi Flower. Validation set dan test set dipertahankan secara
  terpusat sebagai penyederhanaan simulasi. Penelitian tidak mencakup
  implementasi lintas institusi secara fisik, pengukuran latensi jaringan riil,
  maupun evaluasi kegagalan perangkat yang heterogen. Client diasumsikan
  mengikuti protokol pelatihan sedangkan serangan Byzantine, backdoor, dan model
  poisoning tidak diuji. Eksperimen utama mencakup 96 konfigurasi dengan seed 42
  sehingga hasil performa belum mencakup variasi antarseed pelatihan.
+ *Batasan Mekanisme Privasi Tambahan.* Penelitian ini tidak mengintegrasikan
  Differential Privacy (DP), Homomorphic Encryption (HE), maupun Secure
  Multi-Party Computation (SMPC). Pembahasan privasi dibatasi pada mekanisme
  pertukaran informasi dalam rancangan FL, termasuk karakteristik gradient-less
  pada FedXGBllr. Penelitian tidak melakukan pengujian serangan privasi maupun
  pengukuran jaminan privasi formal. Oleh karena itu, pembatasan perpindahan
  data mentah selama pelatihan tidak diartikan sebagai bukti bahwa seluruh model
  atau parameter yang dipertukarkan bebas dari risiko kebocoran informasi.
+ *Batasan Penanganan Class Imbalance.* Teknik penanganan class imbalance yang
  dibandingkan sebagai perlakuan eksperimen adalah SMOTE (Synthetic Minority
  Oversampling Technique) dan tanpa SMOTE. SMOTE diterapkan pada data pelatihan
  masing-masing client untuk skenario federated serta pada training set terpusat
  untuk baseline. Target rasio minoritas terhadap mayoritas ditetapkan sebesar
  1:100 dengan lima tetangga minoritas. Penerapannya dilewati apabila jumlah
  minoritas kurang dari enam atau target rasio telah terpenuhi. Kondisi dengan
  aturan SMOTE karena itu tidak selalu menghasilkan sampel tambahan pada setiap
  client. Teknik lain seperti undersampling, cost-sensitive learning, focal
  loss, dan Adaptive Synthetic Sampling (ADASYN) @he2008adasyn tidak diuji
  sebagai perlakuan pembanding dalam studi ablasi ini.
+ *Batasan Metrik Evaluasi.* Evaluasi performa menggunakan Area Under the
  Precision-Recall Curve (AUPRC) sebagai metrik utama dan Recall\@5%FPR sebagai
  metrik pelengkap untuk mengukur kemampuan deteksi pada batas false positive
  rate tertentu. F1-score, Precision, dan Recall juga dicatat sebagai metrik
  pelengkap. Accuracy tidak dijadikan acuan utama karena kurang merepresentasikan
  kemampuan deteksi kelas minoritas pada data yang tidak seimbang.
+ *Batasan Metode Explainability.* Analisis explainability dibatasi pada SHapley
  Additive exPlanations (SHAP) @lundberg2017shap dengan fokus pada konsistensi
  feature importance antar client dan stabilitasnya pada kondisi Non-IID.
  Pengukuran utama menjelaskan model global akhir yang sama menggunakan
  background lokal dan sampel penjelasan yang sama antar client. Variasi
  estimator KernelSHAP diperiksa melalui pengulangan seed dan pemeriksaan
  dilakukan secara post-hoc dalam lingkungan simulasi. Evaluasi utama tidak
  membandingkan SHAP dengan Local Interpretable Model-agnostic Explanations
  (LIME), Integrated Gradients, atau attention-based explanation.
+ *Batasan Aspek Non-Teknis.* Aspek non-teknis seperti dampak regulasi spesifik,
  biaya implementasi industri, integrasi dengan sistem core banking, serta
  evaluasi kepuasan pengguna akhir (auditor atau analis fraud) tidak dibahas
  dalam penelitian ini. Kontribusi praktis yang dirumuskan bersifat rekomendasi
  metodologis dan implikasi teoretis, bukan rekomendasi implementasi produksi.

== Tujuan

Berdasarkan rumusan masalah yang telah ditetapkan, penelitian ini memiliki tiga
tujuan utama yang saling terkait, yaitu:

+ Mengevaluasi dan membandingkan performa model Federated Learning berbasis tree
  (FedXGBllr) terhadap model berbasis gradient (Logistic Regression, Support
  Vector Machine) dengan agregasi FedAvg, model Gradient Boosting Machine (GBM)
  dengan agregasi best-model selection, serta model deep learning (FFD berbasis
  1D-CNN dan BERT berbasis tabular Transformer) dengan agregasi
  accuracy-weighted FedAvg dalam deteksi financial fraud pada dataset PaySim,
  ULB Credit Card, dan BAF, menggunakan AUPRC dan Recall\@5%FPR sebagai dasar
  utama pembahasan performa.
+ Menganalisis pengaruh kondisi distribusi data IID dan Non-IID berbasis partisi
  Dirichlet serta penerapan Synthetic Minority Oversampling Technique (SMOTE)
  terhadap performa keenam model melalui perbandingan konfigurasi yang tersedia
  pada masing-masing dataset.
+ Menganalisis karakteristik explainability keenam model menggunakan SHapley
  Additive exPlanations (SHAP), khususnya konsistensi feature importance antar
  client dan stabilitas interpretasi di bawah kondisi Non-IID, dengan
  mempertimbangkan variasi estimator pada pengukuran KernelSHAP.

== Manfaat

Penelitian ini diharapkan memberikan manfaat baik secara teoritis maupun praktis
sebagai berikut:

*A. Manfaat Teoritis*

+ Memperkaya literatur Federated Learning dengan menyediakan benchmark
  komparatif yang mencakup empat paradigma agregasi, yaitu FedAvg untuk model
  parametrik, best-model selection untuk GBM, accuracy-weighted FedAvg untuk
  model deep learning, dan tree ensemble aggregation berbasis learnable learning
  rates untuk FedXGBllr dalam satu kerangka eksperimen pada tiga dataset deteksi
  financial fraud.
+ Memberikan bukti empiris mengenai performa FedXGBllr pada data fraud dengan
  class imbalance dan partisi Dirichlet Non-IID sebagai perluasan evaluasi dari
  dataset publik berskala umum yang digunakan pada penelitian asalnya.
+ Memberikan kontribusi awal terhadap kajian Explainable Federated Learning
  melalui analisis stabilitas interpretasi SHAP antar client dan antar paradigma
  agregasi yang masih jarang ditelaah dalam literatur.
+ Menjadi dasar metodologis bagi pengembangan kerangka evaluasi yang
  menggabungkan performa deteksi dan stabilitas interpretasi pada model FL
  parametrik, berbasis tree, dan deep learning untuk data tabular, termasuk
  pemeriksaan variasi estimator SHAP sebelum menarik kesimpulan mengenai
  perbedaan antar client.

*B. Manfaat Praktis*

+ Bagi industri perbankan dan lembaga jasa keuangan, penelitian ini memberikan
  pertimbangan berbasis hasil eksperimen mengenai pemilihan konfigurasi model
  dan agregasi FL untuk deteksi fraud kolaboratif. Pertimbangan tersebut
  mencakup performa, pengaruh heterogenitas data, dan konsistensi interpretasi,
  sebagai dasar kajian lanjutan sebelum penerapan pada lingkungan operasional.
+ Bagi otoritas regulator seperti Otoritas Jasa Keuangan (OJK) dan Pusat
  Pelaporan dan Analisis Transaksi Keuangan (PPATK), hasil penelitian ini dapat
  menjadi referensi teknis dalam menyusun pedoman pemanfaatan teknologi
  Federated Learning untuk pengawasan dan deteksi transaksi mencurigakan,
  terutama dalam menjawab kebutuhan akan model yang akurat sekaligus transparan
  dan dapat diaudit.
+ Bagi pengembang sistem deteksi fraud, penelitian ini menyediakan panduan
  praktis mengenai trade-off antara performa, ketahanan terhadap heterogenitas
  data, dan interpretabilitas model dalam memilih arsitektur FL yang tepat
  sesuai konteks operasional, termasuk implikasi penggunaan SMOTE pada skema
  federated.
+ Bagi komunitas akademik, hasil eksperimen dan implementasi penelitian ini
  dapat menjadi baseline yang direplikasi untuk pengembangan riset lanjutan,
  baik dalam pengujian skema agregasi alternatif (FedProx, SCAFFOLD), integrasi
  mekanisme privasi tambahan (Differential Privacy), maupun ekstensi ke domain
  keuangan lainnya.

= TINJAUAN PUSTAKA

== Hasil Penelitian Terdahulu

=== Deteksi Financial Fraud Berbasis Machine Learning dan Deep Learning Terpusat

Penelitian deteksi fraud secara terpusat telah berkembang pesat selama satu
dekade terakhir yang terpusat pada arsitektur deep learning.
#cite(<sharma2022>, form: "prose") mengembangkan pendekatan credit card fraud
detection berbasis Auto-Encoder yang digabungkan dengan klasifikasi deep neural
network. Auto-encoder digunakan untuk mempelajari representasi laten dari pola
transaksi normal sedangkan transaksi yang menunjukkan rekonstruksi dengan error
tinggi diklasifikasikan sebagai anomali. Pendekatan ini berhasil meningkatkan
kemampuan deteksi pada data dengan ketidakseimbangan kelas namun tetap dalam
kondisi terpusat seluruh data transaksi dapat dikumpulkan pada satu titik
komputasi.

Pengembangan lebih lanjut dilakukan oleh #cite(<baghdadi2024>, form: "prose")
yang mengusulkan pendekatan ensemble learning yang menggabungkan energy-based
Restricted Boltzmann Machine (RBM) dengan extended Long Short-Term Memory
(xLSTM) untuk predictive analytics pada deteksi fraud kartu kredit. Hasil
eksperimen menunjukkan bahwa kombinasi pembelajaran representasi berbasis energi
dengan arsitektur sekuensial mampu menangkap pola temporal yang kompleks pada
transaksi finansial. Meskipun kedua penelitian tersebut menunjukkan keunggulan
performa, keduanya beroperasi dalam kondisi terpusat sehingga tidak menjawab
persoalan privasi dan kepatuhan regulasi yang menjadi penting dalam konteks
lintas institusi. Selain itu, model-model berbasis deep learning tersebut
bersifat black-box sehingga aspek interpretabilitas menjadi sulit dipenuhi tanpa
metode XAI tambahan.

=== Penerapan Federated Learning untuk Deteksi Financial Fraud

Sebagai jawaban atas keterbatasan kondisi terpusat atau centralized, sejumlah
peneliti mulai mengeksplorasi penerapan Federated Learning untuk deteksi
financial fraud. #cite(<suvarna2020>, form: "prose") merupakan salah satu
kontributor awal yang mendemonstrasikan penerapan FL pada credit card fraud
detection. Penelitian tersebut menggunakan model konvensional yang dilatih
secara federated dengan FedAvg dan menunjukkan bahwa pendekatan FL mampu
memberikan performa yang sebanding dengan pendekatan terpusat dengan tetap
menjaga privasi data pelanggan. Namun, penelitian tersebut belum
mempertimbangkan kompleksitas distribusi data Non-IID yang merupakan kondisi
nyata pada kolaborasi antar institusi keuangan.

Penelitian yang lebih lanjut oleh #cite(<venkatakrishna2024>, form: "prose")
memperluas eksplorasi tersebut dengan mengintegrasikan arsitektur deep learning
ke dalam kerangka FL untuk deteksi fraud kartu kredit. Pendekatan ini berhasil
meningkatkan akurasi deteksi melalui pemanfaatan kapasitas representasi deep
neural network namun masih memiliki kelemahan model deep learning dalam hal
interpretabilitas dan tetap mengandalkan agregasi berbasis FedAvg yang sensitif
terhadap heterogenitas data.

Perkembangan paling menjanjikan datang dari #cite(<tang2024>, form: "prose")
yang mengusulkan kerangka federated graph learning untuk deteksi fraud kartu
kredit. Penelitian tersebut memanfaatkan struktur graf untuk memodelkan hubungan
antar entitas transaksi, kemudian melatih Graph Neural Network (GNN) secara
federated di atas kerangka tersebut. Hasilnya menunjukkan bahwa pendekatan
berbasis graf mampu menangkap pola fraud yang bersifat relasional yang sulit
dideteksi oleh model konvensional. Walaupun demikian, pendekatan ini menambahkan
kompleksitas komputasi yang signifikan dan tetap menggunakan paradigma agregasi
berbasis gradient sehingga belum mengeksplorasi kemungkinan penggunaan model
berbasis tree yang justru terbukti unggul untuk data tabular transaksional.

#cite(<aljunaid2025>, form: "prose") memberikan kontribusi penting dengan
mengusulkan kerangka FL berbasis XAI untuk deteksi fraud perbankan. Penelitian
tersebut menggunakan tiga model konvensional, yaitu Logistic Regression (LR),
Support Vector Machine (SVM), dan Gradient Boosting Machine (GBM) dengan skema
agregasi best-model selection, yaitu pemilihan model dengan akurasi terbaik di
antara seluruh client. Hasilnya menunjukkan bahwa model GBM mencapai performa
terbaik dan integrasi SHAP berhasil memberikan transparansi terhadap keputusan
model. Meskipun pendekatan ini telah memperkenalkan dimensi explainability ke
dalam FL, penelitian tersebut belum membandingkan hasilnya dengan model FL
berbasis tree yang lebih modern seperti FedXGBllr, dan belum mengkaji secara
spesifik bagaimana skema agregasi yang berbeda memengaruhi stabilitas
interpretasi SHAP di bawah kondisi Non-IID.

=== Integrasi Model Berbasis Tree ke dalam Kerangka Federated Learning

Eksplorasi model berbasis tree dalam ekosistem FL relatif terbatas dibandingkan
model berbasis gradient. Hal ini disebabkan oleh karakteristik struktur pohon
berupa aturan percabangan yang dapat berbeda antar client sehingga skema
agregasi standar seperti FedAvg tidak dapat diterapkan secara langsung pada
struktur tersebut. #cite(<ma2023fedxgbllr>, form: "prose") mengusulkan FedXGBllr
(Federated XGBoost with Learnable Learning Rates), sebuah kerangka pelatihan
XGBoost secara federated dalam horizontal setting yang tidak bergantung pada
pertukaran gradient dan hessian antar client. Setiap client melatih tree
ensemble secara lokal, kemudian server mengagregasi seluruh tree ensemble dan
melatih one-layer 1D Convolutional Neural Network (CNN) untuk mempelajari
learning rate setiap pohon secara global. Pendekatan ini terbukti menurunkan
communication overhead hingga 25--700 kali lipat dibandingkan metode sebelumnya
seperti SimFL sekaligus menghilangkan risiko kebocoran privasi melalui gradient.

Pada tahap berikutnya, output dari kumpulan pohon tersebut menjadi input bagi
one-layer 1D Convolutional Neural Network (CNN) untuk mempelajari kontribusi
pohon terhadap prediksi akhir. Dengan demikian, komponen CNN pada FedXGBllr
bekerja atas representasi output pohon dan bukan atas urutan transaksi nasabah.
Pendekatan ini memungkinkan penggabungan informasi dari tree ensemble lokal
tanpa merata-ratakan struktur pohon secara langsung @ma2023fedxgbllr.

Walaupun demikian, evaluasi original FedXGBllr dilakukan pada dataset publik
berskala umum seperti HIGGS, SUSY, dan a9a yang tidak merefleksikan
karakteristik domain finansial. Dataset-dataset tersebut tidak memiliki class
imbalance ekstrem maupun struktur fitur transaksional yang khas pada deteksi
fraud. Selain itu, evaluasi original menggunakan partisi data yang seimbang dan
tidak menguji performa pada skenario Non-IID berbasis distribusi Dirichlet yang
lebih realistis. Aspek explainability model juga belum dieksplorasi dalam
penelitian #cite(<ma2023fedxgbllr>, form: "prose") dimana FedXGBllr memiliki
potensi interpretasi melalui struktur pohon yang transparan dan bobot learning
rate yang dapat dipelajari.

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
bahwa SHAP mampu memberikan feature importance yang konsisten dan dapat diaudit
sehingga cocok untuk diadopsi pada domain finansial yang ketat regulasinya.
Penelitian ini menjadi pijakan penting bahwa kombinasi model berbasis tree
dengan SHAP merupakan pasangan yang efektif untuk skenario keuangan.

#cite(<aljunaid2025>, form: "prose") telah memperkenalkan integrasi XAI ke dalam
kerangka FL untuk deteksi fraud perbankan. Namun, analisis SHAP yang mereka
lakukan masih bersifat agregat dan belum menganalisis bagaimana variasi
distribusi data antar client memengaruhi konsistensi feature importance. Oleh
karena itu, penelitian ini menganalisis stabilitas interpretasi SHAP pada
kondisi IID dan Non-IID dengan membandingkan model parametrik, model berbasis
tree, dan model deep learning. Perbandingan tersebut mencakup FedAvg untuk LR
dan SVM, best-model selection untuk GBM, accuracy-weighted FedAvg untuk FFD dan
BERT, serta tree ensemble aggregation dengan learnable learning rates untuk
FedXGBllr.

=== Rangkuman Penelitian Terdahulu

Berdasarkan kajian di atas, penelitian ini menganalisis penerapan FL untuk
deteksi financial fraud dengan class imbalance ekstrem dan distribusi Non-IID,
integrasi model berbasis tree, khususnya FedXGBllr yang bersifat gradient-less,
ke dalam ekosistem FL untuk sektor keuangan, dan stabilitas interpretasi SHAP
antar paradigma agregasi FL. @tab-2-1 merangkum hasil penelitian terdahulu yang
dikaji beserta identifikasi celah riset yang menjadi dasar penelitian ini.

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
kredit, money laundering, identity theft, dan transaksi yang mencurigakan.
Karakteristik utama yang membedakan deteksi fraud dari permasalahan klasifikasi
lainnya adalah ketidakseimbangan kelas yang ekstrem (extreme class imbalance) di
mana proporsi transaksi fraud terhadap transaksi normal seringkali kurang dari
1%.

Pendekatan deteksi fraud secara umum dapat dikelompokkan menjadi tiga kategori.
Pendekatan rule-based menggunakan aturan yang ditetapkan oleh ahli domain namun
kurang adaptif terhadap pola fraud baru. Pendekatan machine learning
memanfaatkan algoritma pembelajaran untuk mengidentifikasi pola dari data
historis transaksi. Adapun pendekatan deep learning menggunakan arsitektur
jaringan saraf dalam untuk menangkap pola non-linear yang kompleks. Penelitian
ini membandingkan model parametrik, model berbasis tree, dan model deep learning
untuk mengamati perbedaan kemampuan deteksi serta karakteristik interpretasinya
pada data tabular finansial.

=== Federated Learning

Federated Learning (FL) adalah paradigma pembelajaran mesin terdistribusi yang
memungkinkan beberapa pihak (client) melatih model bersama tanpa memindahkan
data mentah dari masing-masing pemiliknya @mcmahan2023fedavg. Pelatihan
dilakukan secara lokal di setiap client, dan hanya parameter atau pembaruan
model yang dikomunikasikan ke server pusat untuk diagregasi menjadi model
global. Mekanisme tersebut membatasi perpindahan data mentah selama pelatihan,
meskipun pertukaran parameter atau model tidak dengan sendirinya memberikan
jaminan privasi terhadap seluruh bentuk serangan.

#figure(
  image("resources/fig-2-1-fl-architecture.png", width: 80%),
  caption: [Arsitektur umum Federated Learning. Sumber: #cite(<sabuhi2024microfl>, form: "prose")],
) <fig-2-1>

Berdasarkan distribusi data, FL diklasifikasikan menjadi tiga jenis utama, yaitu
Horizontal Federated Learning (HFL) ketika client memiliki ruang fitur yang sama
tetapi sampel yang berbeda, Vertical Federated Learning (VFL) ketika client
memiliki ruang sampel yang sama tetapi fitur yang berbeda, dan Federated
Transfer Learning (FTL) ketika ruang fitur dan ruang sampel berbeda. Penelitian
ini menggunakan paradigma HFL karena setiap institusi keuangan diasumsikan
memiliki struktur fitur transaksi yang sama tetapi sampel nasabah yang berbeda.

Berdasarkan skala dan jumlah client, FL juga dibedakan menjadi cross-device FL
(jutaan perangkat seperti telepon genggam) dan cross-silo FL (puluhan organisasi
seperti bank atau rumah sakit). Penelitian ini berada pada konteks cross-silo FL
dengan asumsi honest-but-curious clients, yaitu client mengikuti protokol
pelatihan dengan jujur namun mungkin mencoba menyimpulkan informasi dari pesan
yang diterima. Proses pelatihan FL secara umum mengikuti empat tahapan iteratif
@kairouz2021:

+ Inisialisasi: Server menginisialisasi model global $w_0$ dan
  mendistribusikannya kepada seluruh client.
+ Pelatihan Lokal: Setiap client $k$ melatih model menggunakan data lokalnya
  $D_k$ untuk menghasilkan model lokal $w_t^k$.
+ Agregasi: Client mengirimkan parameter atau pembaruan model ke server, yang
  kemudian mengagregasi seluruh kontribusi menjadi model global $w_(t+1)$.
+ Iterasi: Model global terbaru dikirim kembali ke client untuk putaran
  berikutnya hingga konvergensi tercapai atau jumlah putaran maksimum terpenuhi.

Adapun Federated Averaging (FedAvg) yang diperkenalkan oleh
#cite(<mcmahan2023fedavg>, form: "prose") merupakan algoritma agregasi paling
fundamental dalam FL. FedAvg melakukan rata-rata berbobot terhadap parameter
model dari seluruh client berdasarkan ukuran data lokal masing- masing,
sebagaimana dirumuskan pada @eq-fedavg:

$ w_(t+1) = sum_(k=1)^K n_k / n w_t^k $ <eq-fedavg>

dengan $w_(t+1)$ sebagai parameter global pada putaran ke-$(t + 1)$, $K$ sebagai
jumlah client, $n_k$ sebagai jumlah sampel pada client ke-$k$, $n = sum n_k$
sebagai total sampel, dan $w_t^k$ sebagai parameter lokal client ke-$k$ pada
putaran ke-$t$. FedAvg cocok untuk model parametrik yang dilatih dengan gradient
descent seperti Logistic Regression dan Support Vector Machine namun tidak dapat
diterapkan secara natural untuk model berbasis tree karena struktur pohon tidak
dapat dirata-ratakan secara element-wise.

Untuk model yang tidak kompatibel dengan agregasi berbobot seperti FedAvg,
#cite(<aljunaid2025>, form: "prose") memperkenalkan skema best-model selection
yang merumuskan agregasi sebagai pemilihan model dengan kinerja terbaik di
antara seluruh client, sebagaimana dirumuskan pada @eq-bestmodel:

$ W^* = op("arg max", limits: #true)_(W_i) A(W_i, V_i) $ <eq-bestmodel>

dengan $W^*$ sebagai bobot model global terpilih, $W_i$ sebagai bobot model dari
client ke-$i$, $A(dot)$ sebagai fungsi evaluasi kinerja (misalnya akurasi atau
AUPRC), dan $V_i$ sebagai validation set. Skema ini cocok untuk model Gradient
Boosting Machine (GBM) yang strukturnya tidak dapat dirata-ratakan dengan
trade-off berupa penyederhanaan agregasi dan potensi kehilangan informasi dari
client yang tidak terpilih.

Selain ukuran data lokal, kontribusi client dalam agregasi dapat juga
mempertimbangkan kinerja model lokal. Pada varian accuracy-weighted FedAvg yang
digunakan dalam penelitian ini, AUPRC lokal digunakan sebagai komponen
pembobotan bersama ukuran data client. Dengan demikian, istilah accuracy-weighted
pada nama skema tidak merujuk pada penggunaan metrik accuracy sebagai bobot.
Penyesuaian ini relevan pada data dengan class imbalance karena accuracy yang
tinggi masih dapat diperoleh oleh model yang gagal mendeteksi kelas fraud.

Perbandingan antar paradigma agregasi juga perlu mempertimbangkan model yang
digunakan. FedAvg, best-model selection, dan agregasi tree ensemble diterapkan
pada struktur model yang berbeda. Maka dari itu, perbedaan performa antar
kelompok dalam penelitian ini menggambarkan konfigurasi model dan agregasinya
secara bersamaan sehingga tidak seluruhnya dapat diatribusikan pada paradigma
agregasi saja.

=== Model Parametrik (Gradient-Based)

Logistic Regression adalah model klasifikasi linear yang memodelkan probabilitas
keluaran biner menggunakan fungsi sigmoid. Model ini bekerja dengan mempelajari
sebuah vektor bobot $w$ yang merepresentasikan pengaruh setiap fitur terhadap
kemungkinan kelas tertentu. Hasil kombinasi linear antara fitur masukan dan
bobot kemudian ditransformasikan oleh fungsi sigmoid agar menghasilkan
probabilitas pada rentang 0 hingga 1.

Pelatihan LR dilakukan dengan meminimalkan binary cross-entropy loss dan
parameternya berupa vektor koefisien serta bias sehingga kompatibel dengan
agregasi FedAvg. Implementasi penelitian ini menggunakan solver lbfgs yang
merupakan metode quasi-Newton bermemori terbatas dan bukan gradient descent.
Keluaran pelatihannya tetap berupa koefisien linear yang dapat dirata-ratakan
antar client. Keunggulan utama LR adalah interpretabilitas koefisien yang
langsung menunjukkan arah dan besarnya pengaruh setiap fitur. Namun, LR memiliki
keterbatasan dalam menangkap interaksi non-linear antar fitur sehingga kurang
optimal untuk pola fraud yang kompleks.

Adapun Support Vector Machine (SVM) merupakan algoritma klasifikasi yang mencari
hyperplane pemisah optimal dengan margin maksimum antara dua kelas
@cortes1995svm. Secara intuitif, SVM berusaha menggambar garis pemisah yang
sejauh mungkin darititik-titik data terdekat di kedua kelas, sehingga model
lebih robust terhadap data baru.

Pada pelatihan SVM, terdapat parameter regularisasi yang mengontrol trade-off
antara lebar margin dan toleransi terhadap kesalahan klasifikasi. Untuk data
yang tidak dapat dipisahkan secara linear, SVM dapat diperluas dengan kernel
trick yang memetakan data ke ruang berdimensi lebih tinggi. Pada konteks
penelitian ini, SVM linear digunakan agar parameternya dapat diagregasi
menggunakan FedAvg.

=== Model Berbasis Tree

Gradient Boosting Machine (GBM) adalah model ensemble yang membangun pohon
keputusan secara aditif dan berurutan dengan setiap pohon dilatih untuk
memperbaiki kesalahan pohon sebelumnya. Prediksi akhir model adalah jumlah
berbobot dari prediksi seluruh pohon dalam ensemble dengan kontribusi setiap
pohon dikontrol oleh learning rate. Mekanisme ini memungkinkan GBM membangun
model yang kuat dari banyak weak learner berupa pohon- pohon dangkal.

GBM unggul dalam menangani fitur heterogen, interaksi non-linear, dan nilai
missing sehingga sangat cocok untuk data tabular transaksional. Namun, struktur
pohonnya yang berupa urutan kondisi percabangan tidak dapat dirata-ratakan
secara langsung sehingga GBM memerlukan skema agregasi alternatif seperti
best-model selection dalam konteks FL.

Extreme Gradient Boosting (XGBoost) merupakan implementasi GBM yang dioptimalkan
oleh #cite(<chen2016xgboost>, form: "prose") dengan penambahan regularisasi
eksplisit, dukungan komputasi paralel, dan penanganan missing value yang
efisien. XGBoost menambahkan komponen regularisasi pada fungsi loss-nya untuk
mengontrol kompleksitas pohon sehingga model lebih tahan terhadap overfitting
dibandingkan GBM klasik.

Untuk membangun setiap pohon, XGBoost menggunakan informasi turunan pertama dan
turunan kedua dari fungsi loss untuk mengevaluasi kualitas setiap kandidat split
di tiap node. Pendekatan berbasis turunan inilah yang membuat XGBoost sangat
efisien namun sekaligus menjadi tantangan utama saat hendak diintegrasikan ke
dalam FL karena dalam skema federated tradisional, client harus saling bertukar
informasi turunan tersebut, yang berimplikasi pada frekuensi komunikasi yang
sangat tinggi dan potensi kebocoran privasi karena informasi turunan dapat
dieksploitasi untuk merekonstruksi data pelatihan @zhu2019deepleakage.

FedXGBllr yang diusulkan oleh #cite(<ma2023fedxgbllr>, form: "prose") merupakan
kerangka horizontal federated XGBoost yang dirancang untuk mengatasi
keterbatasan XGBoost konvensional dalam FL. Inovasi utamanya terletak pada sifat
gradient-less dimana client tidak perlu bertukar informasi turunan apapun
melainkan hanya mengirimkan tree ensemble yang sudah jadi.

#figure(
  image("resources/fig-2-2-fedxgbllr-architecture.jpg", width: 90%),
  caption: [Arsitektur FedXGBllr. Sumber: #cite(<ma2023fedxgbllr>, form: "prose"). (a) tahap tree ensemble aggregation dari seluruh client dan (b) struktur one-layer 1D CNN untuk mempelajari learning rate setiap pohon.],
) <fig-2-2>

Mekanismenya terdiri dari dua tahap utama. Tahap pertama, setiap client melatih
XGBoost tree ensemble lokal menggunakan datanya sendiri kemudian mengirimkan
seluruh tree ensemble tersebut ke server. Server lalu menggabungkan seluruh tree
ensemble dari semua client menjadi satu kumpulan pohon agregat yang besar.

Tahap kedua, server tidak hanya menggabungkan prediksi pohon-pohon tersebut
secara naif melainkan mempelajari bobot kontribusi (learning rate) untuk setiap
pohon melalui sebuah one-layer 1D Convolutional Neural Network (CNN) yang
dilatih secara federated dengan FedAvg. Dengan kata lain, FedXGBllr membiarkan
setiap client berkontribusi melalui struktur pohonnya namun seberapa besar
pengaruh setiap pohon terhadap prediksi akhir ditentukan secara adaptif oleh CNN
tersebut.

Karakteristik gradient-less pada FedXGBllr menghilangkan kebutuhan pertukaran
informasi turunan dalam pembentukan ensemble tetapi tidak dapat diartikan
sebagai jaminan bahwa model yang dipertukarkan bebas dari seluruh risiko
kebocoran informasi. Selain itu, jumlah putaran komunikasi tidak bergantung pada
kedalaman atau jumlah pohon sehingga communication overhead berkurang secara
signifikan hingga 25--700 kali lebih efisien dibandingkan metode FL berbasis
tree sebelumnya seperti SimFL.

=== Model Deep Learning

Model deep learning mempelajari representasi fitur melalui beberapa lapisan
transformasi non-linear. Pada data tabular, struktur masukan tersebut berbeda
dari citra maupun teks karena setiap kolom dapat memiliki arti dan skala yang
berbeda. Oleh karena itu, arsitektur model perlu disesuaikan dengan bentuk fitur
yang digunakan dalam penelitian.

Model FFD pada penelitian ini menggunakan 1D Convolutional Neural Network.
Operasi konvolusi diterapkan pada representasi fitur transaksi untuk membentuk
pola yang kemudian digunakan dalam klasifikasi fraud. Penggunaan konvolusi satu
dimensi tersebut tidak menunjukkan bahwa penelitian memodelkan urutan waktu
transaksi. Prediksi tetap diberikan pada setiap baris data tabular yang menjadi
masukan model.

Adapun model yang diberi nama BERT pada eksperimen merupakan tabular Transformer
berbasis FT-Transformer dan bukan model bahasa BERT yang dilatih pada dataset
teks. FT-Transformer mengembangkan representasi token dari fitur tabular dan
menggunakan self-attention untuk mempelajari hubungan antar fitur
@gorishniy2021fttransformer. Nama BERT dipertahankan agar konsisten dengan
identitas model dalam hasil eksperimen sedangkan istilah FT-Transformer
digunakan untuk menjelaskan arsitekturnya.

=== Distribusi Data Non-IID dan Partisi Dirichlet

Pada lingkungan FL, data antar client umumnya bersifat non-independent and
identically distributed (Non-IID) yang berarti setiap client memiliki distribusi
data yang berbeda. Kondisi ini mencerminkan situasi nyata pada institusi
keuangan dimana setiap bank memiliki segmentasi nasabah, profil risiko, dan pola
transaksi yang khas. Heterogenitas data antar client terbukti menurunkan
performa konvergensi model FL terutama pada algoritma agregasi seperti FedAvg
yang mengasumsikan distribusi data relatif seragam @li2021noniid.

Untuk mensimulasikan kondisi Non-IID secara terkontrol dan dapat direplikasi,
partisi berbasis distribusi Dirichlet umum digunakan dalam literatur FL
@hsu2019. Secara intuitif, distribusi Dirichlet dengan parameter konsentrasi
$alpha$ mengontrol seberapa heterogen distribusi label antar client. Nilai
$alpha$ kecil menghasilkan distribusi yang sangat heterogen di mana setiap
client cenderung hanya memiliki sampel dari beberapa kelas saja sehingga
merepresentasikan kondisi Non-IID yang ekstrem. Sebaliknya, nilai $alpha$ besar
menghasilkan distribusi yang mendekati IID di mana proporsi setiap kelas relatif
serupa antar client.

#figure(
  image("resources/fig-2-3-dirichlet-alpha.jpg", width: 90%),
  caption: [Visualisasi pengaruh parameter $alpha$ pada distribusi label antar client. Sumber: #cite(<hsu2019>, form: "prose").],
) <fig-2-3>

=== Synthetic Minority Oversampling Technique (SMOTE)

SMOTE merupakan teknik penanganan class imbalance yang menghasilkan sampel
sintetis dari kelas minoritas untuk menyeimbangkan distribusi kelas
@chawla2002smote. Berbeda dengan teknik oversampling sederhana yang hanya
menduplikasi sampel minoritas, SMOTE membuat sampel baru dengan cara melakukan
interpolasi antara satu sampel minoritas dan tetangga terdekatnya di ruang
fitur. Hasilnya adalah sampel sintetis yang berada di antara dua sampel asli,
sehingga lebih variatif dibandingkan sekadar duplikasi.

Pada penelitian ini, SMOTE diterapkan secara lokal pada setiap client sebelum
proses pelatihan federated dimulai, agar prinsip privasi FL tetap terjaga.
Penerapan SMOTE secara global akan mengharuskan agregasi data mentah ke satu
titik komputasi, yang melanggar paradigma FL.

#figure(
  image("resources/fig-2-4-smote-illustration.jpg", width: 70%),
  caption: [Ilustrasi mekanisme SMOTE. Sumber: #cite(<chawla2002smote>, form: "prose").],
) <fig-2-4>

Efektivitas SMOTE bergantung pada sebaran sampel kelas minoritas di ruang fitur.
Pembentukan sampel melalui interpolasi didasarkan pada anggapan bahwa area di
antara sampel minoritas yang berdekatan dapat merepresentasikan kelas yang sama.
Namun, SMOTE tidak secara khusus membedakan sampel yang mewakili pola umum kelas
minoritas dari outlier. Apabila interpolasi melibatkan outlier atau melewati
wilayah kelas mayoritas, sampel sintetis yang dihasilkan berpotensi memperbesar
tumpang tindih antarkelas dan menyulitkan pembentukan batas keputusan. Oleh
karena itu, penambahan sampel minoritas melalui SMOTE tidak selalu menghasilkan
peningkatan performa klasifikasi karena manfaatnya turut ditentukan oleh
struktur kelas pada data.

Proporsi kelas minoritas sendiri belum sepenuhnya menggambarkan kesulitan
klasifikasi, karena dua dataset dengan rasio fraud yang serupa dapat memiliki
tingkat tumpang tindih kelas yang berbeda.
#cite(<napierala2016types>, form: "prose") membedakan sampel minoritas menjadi
safe, borderline, rare, dan outlier berdasarkan lingkungan lokalnya. Tipologi
ini digunakan untuk memeriksa struktur kesulitan data yang tidak terlihat dari
prevalensi kelas saja: kelas minoritas yang didominasi kategori rare dan outlier
memiliki dukungan lokal yang terbatas sehingga tidak membentuk pola yang
bermakna meskipun jumlah sampelnya besar. Interpretasi tipologi tersebut
bergantung pada representasi fitur dan ukuran jarak yang digunakan sehingga
proporsi kategori diperlakukan sebagai diagnosis terhadap ruang fitur
penelitian, bukan sebagai ukuran kesulitan intrinsik yang terlepas dari
preprocessing.

Selain sebaran kelas, ketersediaan sampel minoritas asli juga perlu
diperhatikan. #cite(<weiss2004rarity>, form: "prose") membedakan kelangkaan
relatif (relative rarity), yaitu rendahnya proporsi suatu kelas dibandingkan
kelas lainnya, dan kelangkaan absolut (absolute rarity), yaitu terbatasnya
jumlah sampel yang tersedia untuk mempelajari karakteristik kelas tersebut.
Dalam konteks SMOTE, perbedaan ini menunjukkan bahwa penambahan jumlah sampel
sintetis tidak dapat disamakan dengan penambahan pengamatan nyata yang
independen. Karena sampel sintetis dibentuk berdasarkan sampel yang telah
tersedia, keragaman pola yang dapat direpresentasikan tetap bergantung pada
cakupan sampel minoritas asli. Dengan demikian, distribusi kelas yang lebih
seimbang setelah oversampling belum menjamin bahwa karakteristik kelas minoritas
telah terwakili secara memadai.

=== Metrik Evaluasi untuk Imbalanced Classification

Pada konteks imbalanced classification, metrik accuracy tidak dapat diandalkan
karena bias terhadap kelas mayoritas. Penelitian ini menggunakan AUPRC sebagai
metrik utama, disertai Recall\@5%FPR untuk mengukur kemampuan deteksi pada batas
false positive rate tertentu. Precision, Recall, dan F1-score digunakan sebagai
metrik pelengkap @saito2015.

Confusion Matrix. Matriks ini menyajikan empat komponen dasar evaluasi: True
Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN),
di mana kelas positif merepresentasikan transaksi fraud.

Precision mengukur proporsi prediksi positif yang benar (@eq-precision):

$ "Precision" = "TP" / ("TP" + "FP") $ <eq-precision>

Recall mengukur proporsi sampel positif yang berhasil dideteksi (@eq-recall):

$ "Recall" = "TP" / ("TP" + "FN") $ <eq-recall>

F1-score merupakan rata-rata harmonik dari Precision dan Recall (@eq-f1):

$ F_1 = 2 dot ("Precision" dot "Recall") / ("Precision" + "Recall") $ <eq-f1>

AUPRC (Area Under the Precision-Recall Curve) merangkum kemampuan model dalam
mendeteksi kelas positif pada berbagai ambang klasifikasi. Perubahan ambang
menghasilkan pasangan nilai Precision dan Recall yang berbeda yang kemudian
membentuk kurva Precision-Recall. Kurva ini menggambarkan hubungan antara
banyaknya kasus positif yang berhasil ditemukan dan ketepatan prediksi positif
model. Dalam penelitian ini, AUPRC dihitung menggunakan average precision, yaitu
penjumlahan nilai Precision yang masing-masing dibobot berdasarkan besarnya
peningkatan Recall. Dengan demikian, model memperoleh nilai yang tinggi apabila
mampu menemukan lebih banyak kasus positif sambil mempertahankan Precision yang
tinggi. Perhitungan ini berbeda dari metode trapezoidal yang menghitung luas
dengan menghubungkan titik-titik kurva menggunakan garis lurus.
#cite(<saito2015>, form: "prose") menunjukkan bahwa AUPRC lebih informatif
dibandingkan AUC-ROC pada data dengan class imbalance ekstrem, karena AUC-ROC
cenderung optimistis ketika kelas negatif jauh lebih banyak dari kelas positif.
Oleh karena itu, AUPRC dipilih sebagai metrik utama dalam penelitian ini.

Recall\@5%FPR mengukur proporsi kasus fraud yang berhasil dideteksi ketika false
positive rate (FPR) berada pada tingkat 5%. FPR merupakan proporsi transaksi sah
yang keliru diklasifikasikan sebagai fraud, sebagaimana ditunjukkan pada
@eq-fpr. Sebagai contoh, nilai Recall\@5%FPR sebesar 0,80 berarti model mampu
mendeteksi 80% kasus fraud dengan tingkat kesalahan penandaan sebesar 5% dari
seluruh transaksi sah.

$ "FPR" = "FP" / ("FP" + "TN") $ <eq-fpr>

Jika AUPRC merangkum performa model pada berbagai ambang klasifikasi,
Recall\@5%FPR menunjukkan kemampuan deteksi pada tingkat kesalahan yang sama.
Metrik ini membantu menilai seberapa banyak fraud yang dapat ditemukan ketika
kesalahan deteksi terhadap transaksi sah dibatasi. Oleh karena itu,
Recall\@5%FPR digunakan sebagai metrik pelengkap untuk membandingkan kemampuan
deteksi model pada kondisi operasional yang lebih spesifik.

Kalibrasi menunjukkan kesesuaian antara probabilitas yang diprediksi model dan
frekuensi kejadian yang sebenarnya. Sebagai contoh, pada kumpulan transaksi
dengan prediksi probabilitas fraud sekitar 20%, model yang terkalibrasi akan
menunjukkan proporsi fraud aktual sekitar 20%. Keluaran berupa probabilitas
belum tentu terkalibrasi sehingga aspek ini perlu dievaluasi tersendiri. Brier
score mengukur rata-rata kuadrat selisih antara probabilitas prediksi dan label
aktual,dengan nilai yang lebih rendah menunjukkan kesalahan prediksi
probabilitas yang lebih kecil. Namun, metrik ini turut dipengaruhi kemampuan
diskriminasi model sehingga bukan ukuran kalibrasi semata. Evaluasi dapat
dilengkapi dengan calibration slope dan calibration-in-the-large
@vancalster2016hierarchy. Slope memiliki nilai ideal 1, nilai di bawah 1
menunjukkan prediksi yang terlalu ekstrem, sedangkan nilai di atas 1 menunjukkan
prediksi yang kurang bervariasi. Calibration-in-the-large memiliki nilai ideal 0,
dengan nilai negatif menunjukkan kecenderungan probabilitas terlalu tinggi dan
nilai positif menunjukkan kecenderungan probabilitas terlalu rendah.

Ambang klasifikasi merupakan batas skor yang digunakan untuk menentukan apakah
suatu transaksi dikategorikan sebagai fraud. Perubahan ambang dapat mengubah
Precision, Recall, dan F1-score meskipun skor prediksi model tetap sama.
Penurunan ambang memungkinkan lebih banyak fraud terdeteksi,tetapi juga dapat
meningkatkan jumlah transaksi sah yang keliru ditandai. Sebaliknya, AUPRC
merangkum kurva Precision-Recall pada berbagai ambang sehingga tidak bergantung
pada pemilihan satu ambang tertentu. Oleh karena itu, evaluasi perlu membedakan
kemampuan model dalam mengurutkan transaksi berdasarkan risiko, ketepatan
probabilitas yang dihasilkan, dan hasil klasifikasi pada ambang yang dipilih.

=== Explainable Artificial Intelligence (XAI) dan SHAP

Explainable Artificial Intelligence (XAI) merujuk pada serangkaian metode dan
teknik yang bertujuan membuat keputusan model machine learning dapat dipahami
oleh manusia @doshivelez2017. Pada domain keuangan, explainability tidak hanya
berfungsi sebagai sarana validasi teknis tetapi juga sebagai prasyarat regulasi
dan transparansi terhadap auditor, regulator, dan nasabah.
#cite(<doshivelez2017>, form: "prose") mengusulkan tiga taksonomi evaluasi
explainability, yaitu application-grounded yang melibatkan evaluasi pada
aplikasi nyata oleh praktisi domain, human-grounded yang menggunakan tugas
eksperimental dengan partisipan manusia, dan functionally-grounded yang
menggunakan proksi formal tanpa keterlibatan manusia.

SHAP @lundberg2017shap adalah kerangka untuk interpretasi prediksi model yang
berlandaskan teori permainan kooperatif. Bagi setiap fitur $j$ pada sampel $x$,
Shapley value $phi_j$ mengukur kontribusi rata-rata fitur tersebut terhadap
prediksi model dibandingkan dengan rata-rata prediksi baseline. Nilai Shapley
dirumuskan pada @eq-shapley:

$ phi_j = sum_(S subset.eq F without {j}) (|S|! (|F| - |S| - 1)!) / (|F|!) [f_x (S union {j}) - f_x (S)] $ <eq-shapley>

dengan $F$ sebagai himpunan seluruh fitur, $S$ sebagai subset fitur tanpa fitur
$j$, dan $f_x (S)$ sebagai nilai prediksi ketika fitur dalam $S$ dipertahankan
serta fitur lainnya diperlakukan menurut distribusi referensi. Distribusi
referensi tersebut disebut background distribution. Pemilihannya memengaruhi
nilai dasar dan kontribusi fitur sehingga penjelasan SHAP selalu perlu dibaca
dalam hubungannya dengan referensi yang digunakan.

Komputasi SHAP memerlukan dua himpunan data yang berbeda peran. Yang pertama
adalah background distribution, yaitu sampel referensi yang digunakan untuk
mengaproksimasi nilai model $E[f(z)]$ ketika subset fitur tertentu diasumsikan
tidak teramati. Background distribution secara konseptual merepresentasikan
distribusi data "normal" yang menjadi acuan baseline interpretasi. Yang kedua
adalah explanation data, yaitu himpunan sampel yang akan dijelaskan kontribusi
fiturnya melalui Shapley values $phi_j$. Pemilihan kedua himpunan ini
memengaruhi validitas interpretasi dimana background yang tidak representatif
menghasilkan baseline yang bias sedangkan explanation data yang berbeda antar
konteks evaluasi membuat hasil interpretasi sulit dibandingkan secara langsung.

Properti local accuracy pada SHAP menyatakan bahwa jumlah kontribusi fitur dan
nilai dasar menghasilkan kembali keluaran model yang dijelaskan. Namun,
pemenuhan hubungan penjumlahan tersebut saja belum membuktikan bahwa seluruh
kontribusi fitur telah dihitung secara eksak. Dua himpunan atribusi yang berbeda
masih dapat menghasilkan jumlah yang sama. Pemeriksaan penjelasan karena itu
perlu mempertimbangkan jenis explainer dan variasi estimasinya. Contoh penyajian
atribusi SHAP dalam bentuk summary plot ditunjukkan pada @fig-2-5.

#figure(
  image("resources/fig-2-5-shap-summary-plot.jpg", width: 90%),
  caption: [Contoh visualisasi SHAP dalam bentuk summary plot. Sumber: #cite(<lundberg2017shap>, form: "prose").],
) <fig-2-5>

LinearSHAP digunakan untuk model linear sedangkan TreeSHAP memanfaatkan struktur
model berbasis pohon. Algoritma TreeSHAP diperkenalkan oleh
#cite(<lundberg2019treeshap>, form: "prose") dan menghitung Shapley value pada
model pohon dalam waktu polinomial. Keduanya memberikan pengukuran deterministik
untuk model, masukan, dan referensi yang tetap. Adapun KernelSHAP digunakan
karena penjelasan diberikan terhadap keseluruhan fungsi prediksi model.
KernelSHAP menggunakan regresi berbobot atas koalisi fitur dan dapat melibatkan
pengambilan sampel koalisi ketika seluruh kombinasi tidak dienumerasi.

Pada penjelasan linear interventional, kontribusi fitur mengikuti hubungan
$phi_j (x) = w_j (x_j - mu_j)$ dengan $w_j$ sebagai koefisien model dan $mu_j$
sebagai rata-rata fitur pada background. Dengan demikian, perubahan rata-rata
background dapat mengubah atribusi meskipun koefisien model tetap. Adapun pada
model non-linear, perubahan bentuk distribusi background juga dapat memengaruhi
penjelasan. Perbedaan tersebut perlu dipertimbangkan ketika membandingkan
stabilitas antar keluarga model.

Untuk merangkum penjelasan sejumlah sampel, feature importance dihitung melalui
rata-rata nilai absolut SHAP. Bagi client $c$ dan fitur $j$, nilai tersebut
dinyatakan pada @eq-gcj:

$ g_(c,j) = 1 / N_c sum_(i=1)^(N_c) |phi_(c,j) (x_i)| $ <eq-gcj>

dengan $N_c$ sebagai jumlah sampel yang dijelaskan pada client $c$. Penggunaan
nilai absolut menunjukkan besar kontribusi fitur tanpa membedakan arah
pengaruhnya. Dengan demikian, dua client dapat memiliki feature importance yang
serupa meskipun kontribusi positif dan negatif pada masing-masing sampel tidak
identik.

Pada konteks evaluasi stabilitas interpretasi antar client, perbandingan nilai
SHAP secara langsung kurang tepat karena setiap client memiliki distribusi fitur
lokal yang berbeda di bawah kondisi Non-IID sehingga rentang nilai $|phi_j|$
tidak setara antar client. Sebagai ilustrasi, suatu client yang memiliki
proporsi transaksi fraud lebih tinggi dapat menghasilkan nilai SHAP yang lebih
besar pada fitur tertentu dibandingkan client lain padahal urutan kepentingan
fitur antar keduanya bisa jadi serupa. Untuk menetralkan perbedaan skala semacam
ini, literatur interpretable machine learning merekomendasikan penggunaan metrik
berbasis peringkat (rank-based metrics) yang bekerja pada urutan kepentingan
fitur dan bukan pada nilai absolutnya @doshivelez2017.

Konsistensi peringkat fitur diukur menggunakan korelasi Spearman, yaitu korelasi
antara peringkat dua vektor importance. Setiap client mengurutkan fitur
berdasarkan $g_(c,j)$ dari fitur paling berpengaruh hingga paling tidak
berpengaruh, sehingga setiap fitur memperoleh nilai peringkat. Korelasi Spearman
antara dua vektor peringkat tersebut dirumuskan pada @eq-spearman:

$ rho_s = 1 - (6 sum_(j=1)^d d_j^2) / (d (d^2 - 1)) $ <eq-spearman>

dengan $d_j$ sebagai selisih peringkat fitur ke-$j$ antara dua client yang
dibandingkan, dan $d$ sebagai jumlah fitur. Nilai positif yang mendekati satu
menunjukkan urutan fitur yang semakin serupa, sedangkan nilai negatif
menunjukkan kecenderungan urutan yang berlawanan. Nilai nol menunjukkan tidak
adanya hubungan monoton yang terukur dan bukan bahwa seluruh fitur berbeda.
Fitur dengan nilai yang sama memperoleh peringkat rata-rata. Pada vektor
konstan, korelasi peringkat tidak terdefinisi dan tidak boleh dilaporkan sebagai
kesepakatan sempurna.

Korelasi Spearman dipilih sebagai metrik stabilitas karena tiga alasan. Pertama,
metrik ini menggunakan informasi peringkat penuh (full ranking information) dari
seluruh fitur sehingga memberikan gambaran menyeluruh mengenai kesesuaian
interpretasi pada semua tingkat kepentingan. Kedua, metrik ini scale-invariant
sehingga tetap valid meskipun nilai SHAP berbeda antar client akibat
heterogenitas data. Ketiga, korelasi peringkat memiliki interpretasi statistik
dapat diuji signifikansinya sehingga klaim stabilitas dapat dipertanggungjawabkan
secara statistik.

Selain korelasi tanpa bobot, penelitian ini menggunakan korelasi peringkat
berbobot nilai agar perbedaan pada fitur dengan kontribusi lebih besar mendapat
perhatian lebih tinggi. Ukuran ini dihitung sebagai korelasi Pearson berbobot
atas kedua vektor peringkat. Bobotnya adalah satu vektor tunggal per sel
pengukuran, yaitu rata-rata $|phi|$ setiap fitur atas seluruh vektor importance
client dan seed pada sel tersebut yang dinormalisasi agar berjumlah satu dan
dipakai identik untuk setiap pasangan di dalam sel. Pembobotan ini melengkapi
Spearman tanpa bobot yang memperlakukan seluruh posisi peringkat secara setara.
Kedua ukuran tetap menilai kesamaan peringkat, bukan kesamaan angka SHAP secara
langsung.

Kesamaan himpunan fitur terpenting dapat diukur melalui Jaccard similarity,
yaitu ukuran irisan kedua himpunan dibagi ukuran gabungannya. Jaccard similarity
pada top-$K$ fitur dirumuskan pada @eq-jaccard:

$ J_K = (|T_a (K) inter T_b (K)|) / (|T_a (K) union T_b (K)|) $ <eq-jaccard>

dengan $T_a (K)$ dan $T_b (K)$ masing-masing sebagai himpunan $K$ fitur teratas
pada client $a$ dan client $b$. Berbeda dari korelasi Spearman yang menilai
keseluruhan urutan, Jaccard similarity hanya menilai kesepakatan mengenai
identitas himpunan fitur teratas tanpa memperhatikan urutan internal di antara
fitur-fitur tersebut. Dua client yang sepakat bahwa lima fitur tertentu adalah
yang paling penting akan memperoleh $J_5 = 1$ meskipun urutan internal kelima
fitur tersebut berbeda.

Kedua metrik dilaporkan bersama karena kombinasinya memungkinkan deteksi kondisi
yang tidak dapat ditangkap oleh metrik tunggal. Dua client dapat memiliki
korelasi Spearman yang tinggi namun Jaccard similarity yang rendah apabila
sebagian besar fitur memiliki peringkat yang konsisten tetapi fitur-fitur
teratasnya berbeda. Sebaliknya, dua client dapat memiliki Jaccard similarity
yang tinggi namun korelasi Spearman yang rendah apabila himpunan fitur
teratasnya identik tetapi urutan keseluruhan fiturnya berbeda. Namun,
perbandingan antar dataset juga perlu memperhitungkan peluang kesamaan yang
dipengaruhi jumlah fitur. Untuk itu, digunakan indeks Kuncheva
@kuncheva2007stability. Apabila dua client memilih $k$ fitur dari $M$ fitur dan
memiliki $r$ fitur yang sama, indeks tersebut dinyatakan pada @eq-kuncheva:

$ I_K = (r - k^2 / M) / (k - k^2 / M), quad 0 < k < M $ <eq-kuncheva>

Indeks ini memperhitungkan irisan yang dapat terjadi secara kebetulan. Nilai
satu menunjukkan himpunan fitur yang identik, nilai nol menunjukkan kesamaan
pada tingkat acuan kebetulan, dan nilai negatif menunjukkan irisan yang lebih
rendah daripada acuan tersebut. Nilai satu tidak menyatakan bahwa urutan atau
besarnya kontribusi fitur di dalam himpunan juga identik. Penelitian ini
menggunakan lima fitur terpenting sebagai ringkasan utama, serta kurva terhadap
$k$ untuk memeriksa ketergantungan hasil pada banyaknya fitur yang dipilih
@nogueira2018stability.

Pada KernelSHAP tersampel, perbedaan vektor importance dapat muncul walaupun
model, background, dan sampel yang dijelaskan tidak berubah. Oleh karena itu,
kesepakatan antar client dibandingkan dengan kesepakatan dari pengulangan
estimator di dalam client yang sama. Pengukuran dalam client menggambarkan
variasi estimator pada konfigurasi yang sedang diperiksa. Perbedaan antar client
yang belum dapat dibedakan dari variasi tersebut tidak langsung menunjukkan
bahwa interpretasinya stabil.

Pengujian stabilitas perlu membedakan perbedaan antar client dari variasi
estimator. Uji exchangeability membentuk distribusi pembanding melalui
pertukaran pemasangan vektor importance di bawah hipotesis nol bahwa pemasangan
tersebut dapat dipertukarkan. Seluruh pemasangan dapat dienumerasi ketika jumlah
vektor cukup kecil. Enumerasi memberikan distribusi pengujian yang lengkap untuk
statistik dan masukan tersebut tetapi interpretasinya tetap bergantung pada
asumsi exchangeability.

Karena pengujian dilakukan pada beberapa konfigurasi, nilai p disesuaikan
menggunakan prosedur Benjamini--Hochberg. Prosedur ini ditujukan untuk
mengendalikan false discovery rate pada suatu keluarga pengujian di bawah asumsi
yang sesuai sehingga keputusan tidak hanya bergantung pada nilai p mentah setiap
konfigurasi @benjamini1995fdr. Konfigurasi yang tidak menunjukkan perbedaan
signifikan tetap tidak dapat disimpulkan memiliki interpretasi yang identik.

Perbandingan stabilitas antara IID dan Non-IID dilakukan secara berpasangan
menggunakan uji Wilcoxon signed-rank satu arah. Pasangan dibentuk dari model,
dataset, dan kondisi SMOTE yang sama sehingga arah selisih menunjukkan apakah
stabilitas IID lebih tinggi daripada Non-IID. Uji ini menguji pola selisih
pasangan dengan asumsi yang menyertainya dan hasilnya tidak menggantikan
pengulangan pelatihan pada beberapa random seed.

=== Flower Framework

Flower (Friendly Federated Learning Research Framework) merupakan kerangka kerja
open-source yang dikembangkan oleh #cite(<beutel2022flower>, form: "prose")
untuk simulasi dan implementasi sistem Federated Learning. Flower menyediakan
abstraksi tingkat tinggi yang memungkinkan peneliti mengimplementasikan berbagai
skema agregasi dan model dengan kompatibilitas terhadap backend populer seperti
PyTorch, TensorFlow, dan scikit-learn. Penelitian ini menggunakan Flower sebagai
infrastruktur simulasi FL dengan implementasi mengikuti baseline hfedxgboost
pada repositori resmi Flower.

= METODOLOGI

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
Tahap pertama adalah data acquisition dan preprocessing terhadap dataset PaySim,
ULB Credit Card, dan Bank Account Fraud (BAF). Data dibagi menjadi training set,
validation set, dan test set dengan proporsi 70:15:15, kemudian diproses sesuai
karakteristik fitur masing-masing dataset. Parameter preprocessing yang
dipelajari dari data ditentukan menggunakan training set agar informasi
validation set dan test set tidak digunakan dalam pembentukan transformasi.

Tahap kedua adalah client partitioning, yaitu pembagian training set ke lima
client melalui skema IID atau Non-IID berbasis distribusi Dirichlet dengan
$alpha$ = 0,5. Tahap ketiga adalah local training dan imbalance handling dengan
membandingkan pelatihan tanpa SMOTE dan pelatihan yang menerapkan aturan SMOTE
lokal. Tahap keempat adalah federated aggregation melalui empat paradigma, yaitu
FedAvg untuk LR dan SVM, best-model selection untuk GBM, accuracy-weighted
FedAvg untuk FFD dan BERT, serta tree ensemble aggregation dengan learnable
learning rates untuk FedXGBllr.

Tahap kelima adalah pelaksanaan skenario eksperimen pada kondisi terpusat,
federated IID, dan federated Non-IID. Kondisi terpusat digunakan sebagai
pembanding performa dan bukan sebagai batas atas yang harus selalu melampaui
hasil federated. Tahap keenam adalah evaluation dan explainability analysis yang
mencakup pengukuran performa deteksi, diagnosis karakteristik data, serta
pengukuran konsistensi feature importance antar client menggunakan SHAP.

== Dataset yang Digunakan

Penelitian ini menggunakan tiga dataset deteksi fraud finansial dengan
karakteristik domain yang berbeda untuk menguji generalisasi metode lintas
domain. Dataset utama adalah Financial Fraud Detection Dataset yang merupakan
turunan simulator PaySim, yaitu simulasi transaksi mobile money. Dataset kedua
adalah ULB Credit Card Fraud Detection Dataset, yaitu transaksi kartu kredit
riil yang fiturnya telah dianonimkan melalui Principal Component Analysis (PCA).
Dataset ketiga adalah Bank Account Fraud (BAF), yaitu data sintetis aplikasi
pembukaan rekening bank yang berbeda dengan kedua dataset lainnya karena
menyediakan fitur bernama dan bermakna semantik. PaySim berperan sebagai
benchmark utama karena skala dan karakteristik fiturnya, sementara ULB Credit
Card dan BAF digunakan sebagai dataset pembanding untuk menilai generalisasi
model pada domain fraud yang berbeda. Penggunaan ketiga dataset memungkinkan
perbandingan hasil pada beberapa kondisi data meskipun tidak dengan sendirinya
membuktikan bahwa suatu model dapat digeneralisasikan ke seluruh domain fraud.
Ketiga dataset diproses melalui antarmuka pipeline yang identik sehingga seluruh
model, skema agregasi, dan metrik evaluasi dapat diterapkan tanpa modifikasi.

=== Karakteristik Dataset PaySim

Dataset utama penelitian ini adalah Financial Fraud Detection Dataset yang
dipublikasikan pada platform Kaggle oleh Sriharsha Eedala. Dataset tersebut
merupakan turunan dari simulator PaySim @lopezrojas2016paysim, yaitu simulator
transaksi mobile money yang dikembangkan berdasarkan log transaksi nyata dari
sebuah perusahaan jasa keuangan di Afrika. Dataset ini dipilih karena memenuhi
tiga kriteria yang relevan dengan konteks penelitian, yaitu class imbalance yang
ekstrem dengan rasio fraud sekitar 0,13%, struktur fitur transaksional tabular
yang mewakili karakteristik nyata sektor keuangan, dan skala data yang memadai
untuk simulasi federated dengan beberapa client. Karakteristik utama dataset
disajikan pada @tab-3-1.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Kaggle (Sriharsha Eedala)],
    [Jenis data], [Transaksi mobile money tabular],
    [Jumlah baris], [± 6.362.620 transaksi],
    [Jumlah kolom mentah], [11, termasuk label target],
    [Label target], [isFraud (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 0,13% (kelas minoritas ekstrem)],
    [Tipe fitur], [Numerik dan kategorikal],
  ),
  caption: [Karakteristik Dataset PaySim],
) <tab-3-1>

Deskripsi setiap fitur disajikan pada @tab-3-2

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

=== Karakteristik Dataset ULB Credit Card

Dataset kedua penelitian ini adalah Credit Card Fraud Detection Dataset yang
dipublikasikan pada platform Kaggle oleh Machine Learning Group Université Libre
de Bruxelles (ULB). Dataset tersebut berisi transaksi kartu kredit riil nasabah
di Eropa selama dua hari pada September 2013. Berbeda dengan PaySim yang
bersifat sintetis, dataset ini merepresentasikan pola fraud kartu kredit yang
nyata sehingga berfungsi sebagai pembanding untuk menilai generalisasi model
pada domain fraud yang berbeda. Karakteristik utama dataset disajikan pada
@tab-3-3.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Kaggle (mlg-ulb/creditcardfraud, ULB)],
    [Jenis data], [Transaksi kartu kredit tabular (fitur ter-PCA)],
    [Jumlah baris], [284.807 transaksi],
    [Jumlah kolom mentah], [31 kolom (30 fitur + 1 label)],
    [Label target], [Class (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 0,17% (492 transaksi fraud)],
    [Tipe fitur], [Numerik (Time, Amount, dan V1–V28 hasil PCA)],
  ),
  caption: [Karakteristik Dataset ULB Credit Card],
) <tab-3-3>

Struktur fitur dataset ini berbeda secara fundamental dari PaySim. Fitur V1
hingga V28 merupakan komponen hasil transformasi PCA yang telah dianonimkan
untuk menjaga kerahasiaan informasi nasabah sehingga makna aslinya tidak
dipublikasikan. Komponen PCA bersifat tidak berkorelasi dan terurut menurut
besarnya variansi. Hanya dua fitur yang tersaji dalam skala asli, yaitu Time
(selisih waktu dalam detik terhadap transaksi pertama) dan Amount (nominal
transaksi). Maka dari itu, tahap preprocessing untuk dataset ini jauh lebih
ringkas karena tidak diperlukan pembersihan kolom identifier, one-hot encoding,
maupun feature engineering. Fitur V1--V28 digunakan sebagaimana tersedia dalam
dataset tanpa transformasi tambahan dan fitur Time dan Amount yang dinormalisasi.
Setelah proses ini, dataset menghasilkan vektor berdimensi 30 fitur yang
dikonsumsi oleh seluruh model secara identik dengan pipeline PaySim.

=== Karakteristik Dataset Bank Account Fraud (BAF)

Dataset ketiga penelitian ini adalah Bank Account Fraud (BAF) yang
dipublikasikan oleh Feedzai pada NeurIPS 2022 @jesus2022baf dan tersedia di
platform Kaggle. BAF berisi data aplikasi pembukaan rekening bank daring dengan
label fraud yang menandai aplikasi yang teridentifikasi sebagai penipuan
identitas. Penelitian ini menggunakan varian Base, yaitu data sintetis yang
dibangkitkan berdasarkan data nyata sehingga tidak diperlakukan sebagai
publikasi langsung atas catatan nasabah asli. Karakteristik utama dataset
disajikan pada @tab-baf-char.

#figure(
  kind: table,
  table(
    columns: (5cm, 1fr),
    align: (left, left),
    table.header([*Atribut*], [*Nilai*]),
    [Sumber], [Feedzai, varian Base; diakses melalui Kaggle (sgpjesus/bank-account-fraud-dataset-neurips-2022)],
    [Jenis data], [Data sintetis aplikasi pembukaan rekening bank, dibangkitkan berdasarkan data nyata],
    [Jumlah baris], [1.000.000 aplikasi],
    [Jumlah kolom mentah], [32 kolom (31 fitur + 1 label)],
    [Label target], [fraud_bool (biner: 0 = normal, 1 = fraud)],
    [Rasio fraud], [± 1,10% (11.029 aplikasi fraud)],
    [Tipe fitur], [Numerik dan kategorikal, dengan nama fitur yang bermakna semantik],
    [Dimensi masukan setelah preprocessing], [55 fitur],
  ),
  caption: [Karakteristik Dataset Bank Account Fraud (BAF)],
) <tab-baf-char>

Berbeda dengan PaySim yang memiliki sedikit fitur dan ULB yang fiturnya
teranonimkan melalui PCA, BAF menyediakan fitur bernama yang bermakna secara
semantik, seperti `income`, `customer_age`, `credit_risk_score`,
`proposed_credit_limit`, serta sejumlah fitur velocity dan riwayat alamat.
Kondisi ini menjadikan BAF sebagai dataset yang paling relevan untuk analisis
interpretabilitas berbasis SHAP karena kontribusi setiap fitur dapat ditafsirkan
secara langsung dalam konteks aplikasi pembukaan rekening. Makna semantik
tersebut melekat pada definisi kolomnya dan bukan pada klaim bahwa nilainya
merupakan catatan nasabah yang sebenarnya. Dataset ini memuat lima fitur
kategorikal (`payment_type`, `employment_status`, `housing_status`, `source`,
dan `device_os`) dan selebihnya merupakan fitur numerik.

== Perancangan Sistem

Sistem yang dirancang dalam penelitian ini terdiri dari empat lapisan logis yang
saling terkait, yaitu lapisan data dan preprocessing, lapisan partisi dan client
orchestration, lapisan pelatihan model dengan empat paradigma agregasi FL, dan
lapisan evaluasi yang mencakup pengukuran performa dan analisis explainability.
Arsitektur umum sistem disajikan pada @fig-3-2.

#figure(
  image("resources/fig-3-2-logical-layers.png", height: 42%),
  caption: [Lapisan logis sistem penelitian.],
) <fig-3-2>

=== Perancangan Pembagian Data dan Tahap Preprocessing

Pembagian data dan tahap preprocessing dirancang untuk membentuk representasi
numerik yang sesuai bagi seluruh model sekaligus mendukung perbandingan antara
pelatihan terpusat dan federated learning. Pembagian dilakukan dalam dua
tingkat, yaitu pembagian global menjadi training set, validation set, dan test
set, serta partisi training set ke seluruh client pada skenario federated.
Ketiga dataset, yaitu PaySim, ULB Credit Card, dan BAF, menggunakan proporsi
pembagian dan random seed yang sama dengan stratifikasi berdasarkan label
masing-masing dataset.

Pada tingkat pertama, setiap dataset dibagi menjadi training set sebesar 70%,
validation set sebesar 15%, dan test set sebesar 15%. Pembagian dilakukan
menggunakan stratified sampling agar proporsi kelas fraud tetap terjaga pada
setiap subset dengan random seed sebesar 42 untuk mendukung reproduksibilitas.
Training set digunakan sebagai sumber data pelatihan baik untuk baseline
terpusat maupun model federated. Validation set dipertahankan secara terpusat di
server simulasi dan digunakan untuk pemilihan model pada skema best-model
selection untuk GBM, pemantauan konvergensi serta early stopping pada model yang
menerapkannya, dan penentuan ambang klasifikasi untuk pelaporan F1-score,
Precision, dan Recall. Test set juga dipertahankan secara terpusat dan digunakan
untuk evaluasi akhir setelah model dan ambang klasifikasi ditetapkan. Subset ini
tidak digunakan untuk pelatihan, pemilihan hyperparameter, maupun penentuan
ambang klasifikasi tersebut.

Pembagian global dilakukan sebelum penentuan parameter preprocessing yang
bergantung pada data. Nilai imputasi dan parameter scaling dipelajari hanya dari
training set kemudian digunakan untuk mentransformasikan validation set dan test
set tanpa menghitung ulang parameternya. Ketentuan ini mencegah informasi dari
data validasi dan pengujian memengaruhi pembentukan representasi fitur.
Sementara itu, operasi yang menggunakan aturan tetap seperti penghapusan kolom
yang telah ditentukan dan pembentukan fitur berdasarkan rumus per transaksi
tidak memerlukan estimasi parameter dari keseluruhan dataset.

Pada PaySim, preprocessing mencakup penghapusan kolom `nameOrig` dan `nameDest`
yang merupakan identifier akun dan `isFlaggedFraud` yang merupakan flag dari
aturan deteksi sebelumnya, serta kolom label `isFraud` dipisahkan sebagai target.
Selanjutnya, dilakukan feature engineering dengan membentuk `errorBalanceOrig`
yang didefinisikan sebagai newbalanceOrig − oldbalanceOrg + amount, dan
`errorBalanceDest` yang didefinisikan sebagai oldbalanceDest + amount −
newbalanceDest. Kedua fitur tersebut digunakan untuk merepresentasikan
ketidaksesuaian antara perubahan saldo dan nominal transaksi.

Fitur kategorikal `type` kemudian diubah menggunakan one-hot encoding menjadi
lima kolom biner berdasarkan jenis transaksi. Seluruh matriks fitur
ditransformasikan menggunakan StandardScaler yang di-fit hanya pada training
set. Meskipun model berbasis pohon seperti GBM tidak mensyaratkan scaling,
transformasi yang sama tetap digunakan untuk menjaga konsistensi representasi
masukan lintas model. Hasil dari keseluruhan preprocessing PaySim menghasilkan
13 fitur masukan.

Pada ULB Credit Card, preprocessing tidak mencakup penghapusan identifier,
one-hot encoding, maupun pembentukan fitur turunan. StandardScaler hanya
diterapkan pada fitur Time dan Amount dengan parameter yang dihitung dari
training set. Fitur V1--V28 dipertahankan sebagaimana tersedia dalam dataset
karena telah melalui transformasi PCA oleh penyedia data. Hasil dari keseluruhan
preprocessing ULB menghasilkan 30 fitur masukan.

Pada BAF, preprocessing mencakup penghapusan kolom, encoding fitur kategorikal,
penanganan nilai yang tidak tersedia, dan scaling. Kolom `device_fraud_count`
dihapus karena bernilai konstan nol. Kolom `month` dikeluarkan dari himpunan
fitur dan dipertahankan sebagai informasi pendamping. Keputusan ini membatasi
pemanfaatan informasi bulan yang berkaitan dengan perubahan prevalensi fraud
mengingat pembagian data menggunakan stratified-random split dan bukan pembagian
temporal. Lima fitur kategorikal, yaitu `payment_type`, `employment_status`,
`housing_status`, `source`, dan `device_os` diubah menggunakan one-hot encoding
dengan daftar kategori tetap agar susunan kolom konsisten pada seluruh subset
dan client.

Penanganan nilai yang tidak tersedia pada BAF diterapkan pada lima kolom yang
menggunakan nilai −1, yaitu `prev_address_months_count`, `bank_months_count`,
`current_address_months_count`, `session_length_in_minutes`, dan
`device_distinct_emails_8w`. Pada setiap kolom, ditambahkan indikator biner
missing untuk mempertahankan informasi mengenai ketidaktersediaan nilai. Nilai
−1 kemudian diganti menjadi nilai kosong dan diimputasi menggunakan median dari
training set. Perlu dicatat bahwa `prev_address_months_count` memiliki tingkat
ketidaktersediaan sekitar 71%, sehingga setelah imputasi median kolom tersebut
menjadi hampir konstan dan sebagian besar sinyalnya justru terkandung pada
indikator missing-nya; hal ini merupakan properti data yang diketahui, bukan
anomali. Sebaliknya, nilai negatif pada `intended_balcon_amount`, `velocity_6h`,
dan `credit_risk_score` dipertahankan karena tidak diperlakukan sebagai penanda
data yang tidak tersedia. Setelah encoding dan imputasi, seluruh matriks fitur
ditransformasikan menggunakan StandardScaler yang di-fit hanya pada training
set. Representasi akhir BAF terdiri dari 55 fitur, yaitu 24 fitur numerik, 5
indikator missing, dan 26 kolom hasil one-hot encoding.

Pada tingkat kedua, training set yang telah melalui preprocessing dipartisi ke
seluruh client menggunakan skema IID atau Dirichlet Non-IID. Setiap client $k$
memperoleh subset lokal $D_k$ yang digunakan untuk pelatihan model lokal. Subset
tersebut tidak dipecah lagi menjadi local validation set karena fungsi validasi
telah diakomodasi oleh validation set terpusat. Pada konfigurasi yang
menggunakan SMOTE, pembentukan sampel sintetis dilakukan secara lokal pada $D_k$
setelah partisi client dan sebelum pelatihan. Validation set dan test set tidak
menerima penambahan sampel sintetis.

Penggunaan parameter preprocessing bersama serta validation set dan test set
terpusat merupakan bagian dari rancangan simulasi. Pengaturan ini menyediakan
ruang fitur dan distribusi evaluasi yang konsisten untuk membandingkan keenam
model serta skema agregasinya. Evaluasi akhir mencakup AUPRC, Recall\@5%FPR,
F1-score, Precision, dan Recall, serta evaluasi probabilitas pada model yang
menyediakan keluaran probabilitas. Setelah model akhir ditetapkan, sampel dari
test set juga digunakan sebagai data penjelasan SHAP tanpa memperbarui parameter
model. Dengan demikian, data pelatihan, validasi, dan pengujian memiliki fungsi
yang terpisah sepanjang pipeline penelitian.

=== Perancangan Skema Partisi Client

Partisi client dilakukan terhadap training set yang telah melalui tahap
preprocessing sebagaimana dijelaskan pada Subbab 3.3.1. Sebanyak $K = 5$ client
disimulasikan untuk merepresentasikan institusi yang berpartisipasi dalam
pelatihan federated. Pembagian data dirancang menggunakan dua skema, yaitu
Independent and Identically Distributed (IID) dan Non-IID, untuk mengevaluasi
pengaruh perbedaan distribusi data antar client terhadap performa model.

Pada skema IID, training set dibagi ke seluruh client dengan jumlah sampel yang
relatif merata dan proporsi label yang mendekati distribusi training set secara
keseluruhan. Skema ini digunakan sebagai baseline federated ketika perbedaan
distribusi data antar client relatif kecil. Perbandingannya dengan baseline
terpusat digunakan untuk mengamati perubahan performa akibat pelatihan federated
pada kondisi tersebut. Pada skema Non-IID, pembagian sampel setiap kelas ke
seluruh client ditentukan menggunakan distribusi Dirichlet. Parameter
konsentrasi $alpha$ mengatur tingkat heterogenitas pembagian, dengan nilai yang
lebih kecil cenderung menghasilkan perbedaan proporsi kelas yang lebih besar
antar client. Eksperimen pelatihan utama menggunakan $alpha$ = 0,5 untuk
membentuk kondisi distribusi data yang heterogen. Hubungan antara pembagian
global dan partisi client disajikan pada @fig-3-3.

#figure(
  image("resources/fig-3-3-two-level-split.jpg", width: 85%),
  caption: [Skema pembagian data dua tingkat.],
) <fig-3-3>

Validation set dan test set dipertahankan tanpa perubahan pada baseline
terpusat, federated IID, dan federated Non-IID. Dengan demikian, seluruh
konfigurasi pada dataset yang sama dievaluasi menggunakan sampel yang sama.
Pengaruh skema partisi diamati dengan membandingkan kondisi IID dan Non-IID pada
model serta konfigurasi penanganan class imbalance yang sama. Sementara itu,
perbandingan antar model tetap mempertimbangkan perbedaan algoritma dan aturan
agregasi yang digunakan.

=== Perancangan Skema Class Imbalance Handling

Penanganan class imbalance dirancang dalam dua konfigurasi, yaitu tanpa SMOTE
sebagai baseline dan dengan SMOTE sebagai intervensi yang diuji. Pada skenario
federated, SMOTE diterapkan secara lokal setelah partisi client dan sebelum
pelatihan sehingga pembentukan sampel sintetis tidak memerlukan penggabungan
data mentah antar client. Pada baseline terpusat, aturan yang sama diterapkan
pada training set terpusat. Validation set dan test set tidak menerima
penambahan sampel sintetis agar distribusi kelas pada data evaluasi tetap
dipertahankan.

Ketiga dataset menggunakan parameter `sampling_strategy` sebesar 0,01 dan
`k_neighbors` sebesar 5. Nilai `sampling_strategy` tersebut menetapkan target
rasio jumlah sampel minoritas terhadap mayoritas sebesar 1:100 dan bukan
penyeimbangan penuh menjadi 1:1. Target ini dipilih untuk membatasi jumlah
sampel sintetis yang ditambahkan terutama pada dataset dengan ketimpangan kelas
yang ekstrem. Parameter dipertahankan sama pada seluruh dataset agar
perbandingan menggunakan aturan penanganan yang konsisten tanpa penyetelan
khusus per dataset. Namun, kesamaan parameter tidak berarti jumlah maupun
proporsi penambahan sampel sintetis akan sama karena distribusi kelas awal
berbeda antar dataset dan client.

Penerapan SMOTE diawali dengan pemeriksaan jumlah sampel minoritas dan rasio
kelas pada data pelatihan yang menerima perlakuan. Dengan `k_neighbors` sebesar
5, diperlukan sedikitnya enam sampel minoritas agar setiap sampel memiliki lima
tetangga minoritas selain dirinya sendiri. Oleh karena itu, SMOTE dilewati
apabila jumlah sampel minoritas kurang dari enam. SMOTE juga dilewati apabila
rasio minoritas terhadap mayoritas telah mencapai atau melampaui 0,01. Batas
minimum tersebut merupakan persyaratan pelaksanaan algoritma dan bukan jaminan
bahwa sampel yang tersedia telah cukup untuk merepresentasikan seluruh
karakteristik kelas minoritas.

Pada skenario Non-IID, pemeriksaan dilakukan secara terpisah untuk setiap
client. Akibatnya, konfigurasi dengan SMOTE dapat mencakup client yang menerima
sampel sintetis dan client yang tidak mengalami perubahan data. Status
penerapan, alasan pelewatan, serta jumlah sampel sebelum dan sesudah SMOTE
dicatat untuk menunjukkan perlakuan yang benar-benar diterima setiap client.
Pencatatan ini diperlukan karena konfigurasi dengan SMOTE menunjukkan penggunaan
aturan oversampling dan bukan bahwa seluruh client selalu memperoleh tambahan
sampel.

Pengaruh SMOTE dievaluasi dengan membandingkan konfigurasi dengan dan tanpa
SMOTE pada dataset, model, dan skema partisi yang sama. Pembagian data,
parameter model, serta prosedur evaluasi dipertahankan sama pada pasangan
konfigurasi tersebut. Dengan demikian, perbandingan diarahkan untuk mengamati
perubahan performa yang berkaitan dengan penerapan aturan SMOTE, dengan tetap
memperhatikan status penerapannya pada setiap client.

=== Perancangan Pelatihan Model dan Skema Agregasi

Pelatihan federated dirancang untuk membandingkan enam model, yaitu Logistic
Regression (LR), Support Vector Machine (SVM), Gradient Boosting Machine (GBM),
FFD, BERT, dan FedXGBllr. Setiap model menggunakan skema agregasi yang
disesuaikan dengan bentuk parameter dan arsitekturnya. Empat skema yang
digunakan meliputi FedAvg, best-model selection, accuracy-weighted FedAvg, serta
agregasi ensemble pohon yang dilanjutkan dengan pelatihan parameter pembobotan
melalui jaringan 1D CNN.

Pada LR dan SVM linear, setiap client melatih model menggunakan data lokal
kemudian mengirimkan koefisien dan bias ke server. Server menggabungkan
parameter tersebut menggunakan FedAvg dengan bobot berdasarkan jumlah sampel
pelatihan setiap client. Parameter hasil agregasi kemudian digunakan untuk
melanjutkan pelatihan pada putaran berikutnya. LR menggunakan solver lbfgs,
yaitu metode optimasi quasi-Newton sehingga proses optimasi lokalnya dinyatakan
melalui batas iterasi solver dan bukan local epochs seperti pada pelatihan
jaringan saraf. Meskipun prosedur optimasinya berbeda, parameter linear yang
dihasilkan tetap dapat diagregasi menggunakan rata-rata berbobot.

Pada GBM, setiap client melatih model lokal yang terdiri dari ensemble pohon.
Server kemudian mengevaluasi model-model tersebut menggunakan validation set
terpusat dan memilih model dengan AUPRC tertinggi sebagai model global. Dengan
demikian, agregasi GBM menggunakan best-model selection tanpa merata-ratakan
parameter atau menggabungkan pohon dari seluruh client.

Pada FFD dan BERT, pelatihan lokal dilakukan selama $E$ local epochs sebelum
parameter jaringan dikirimkan ke server. Agregasi menggunakan skema yang dalam
implementasi disebut accuracy-weighted FedAvg dengan bobot yang mempertimbangkan
jumlah sampel lokal dan skor AUPRC masing-masing client. Istilah
accuracy-weighted pada penamaan tersebut merujuk pada pembobotan berdasarkan
performa dan metrik yang digunakan adalah AUPRC, bukan accuracy. Kombinasi kedua
komponen pembobotan ini merupakan rancangan implementasi penelitian.

Pelatihan FedXGBllr dilakukan dalam dua tahap. Pada putaran ke-0, setiap client
membentuk ensemble pohon lokal yang kemudian digabungkan oleh server. Pada
putaran berikutnya, keluaran ensemble tersebut digunakan sebagai masukan bagi
jaringan 1D CNN yang mempelajari pembobotan kontribusi pohon. Parameter jaringan
dilatih secara lokal dan diagregasi menggunakan FedAvg selama putaran 1 hingga
$R$.

Sebagai pembanding, pelatihan terpusat dilakukan menggunakan LR, SVM linear,
GBM, FFD, BERT, dan XGBoost pada training set yang tidak dipartisi ke client.
XGBoost digunakan sebagai pembanding berbasis pohon untuk FedXGBllr, tetapi
tidak memiliki arsitektur gabungan pohon dan CNN yang sama. Oleh karena itu,
perbandingan XGBoost terpusat dengan FedXGBllr federated mencakup perbedaan
arsitektur sekaligus skema pelatihan dan tidak ditafsirkan semata-mata sebagai
pengaruh federasi. Pemetaan model, skema agregasi, dan pembanding terpusat
disajikan pada @tab-3-4.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Model federated*], [*Kategori*], [*Skema Agregasi*], [*Pembanding terpusat*]),
    [Logistic Regression (LR)], [Parametrik], [FedAvg], [LR],
    [Support Vector Machine (SVM)], [Parametrik (linear)], [FedAvg], [SVM linear],
    [Gradient Boosting Machine (GBM)], [Tree ensemble (histogram-based)], [Best-Model Selection], [GBM],
    [FFD], [Deep learning (1D-CNN)], [Accuracy-Weighted FedAvg], [FFD],
    [BERT (FT-Transformer)], [Deep learning (Transformer tabular)], [Accuracy-Weighted FedAvg], [BERT (FT-Transformer)],
    [FedXGBllr], [Tree ensemble + CNN], [Tree Ensemble Aggregation + Learnable LR], [XGBoost],
    )
  ],
  caption: [Pemetaan Model dengan Skema Agregasi Federated Learning dan Model Pembanding Terpusat],
) <tab-3-4>

Seluruh konfigurasi pelatihan utama menggunakan random seed sebesar 42. Batas
maksimum putaran federated ditetapkan sebesar 20 untuk LR, SVM, GBM, FFD, dan
BERT. Pada FedXGBllr, batas maksimum ditetapkan sebesar 20 putaran untuk PaySim
serta 50 putaran untuk ULB Credit Card dan BAF. Jumlah putaran yang diselesaikan
dapat lebih kecil apabila kriteria penghentian terpenuhi. Pemilihan model akhir
menggunakan validation set sedangkan test set digunakan setelah proses pelatihan
dan pemilihan model selesai.

=== Perancangan Modul Evaluasi

Modul evaluasi dirancang untuk mengukur performa prediksi dan karakteristik
explainability model. Evaluasi performa dilakukan menggunakan test set terpusat
yang sama untuk seluruh konfigurasi pada masing-masing dataset. AUPRC digunakan
sebagai metrik utama sedangkan Recall\@5%FPR, F1-score, Precision, dan Recall
digunakan sebagai metrik pelengkap. Seluruh konfigurasi pelatihan utama
dijalankan menggunakan random seed 42 sehingga hasil dilaporkan sebagai satu
nilai per konfigurasi. Pengulangan seed pada analisis SHAP dan pemeriksaan
partisi digunakan untuk mengevaluasi ketidakpastian estimator serta variasi
pembagian data dan bukan sebagai pengulangan pelatihan model.

AUPRC dihitung menggunakan average precision dari skor prediksi model. Untuk
menghitung F1-score, Precision, dan Recall, ambang klasifikasi dipilih
berdasarkan nilai F1 tertinggi pada validation set kemudian diterapkan tanpa
perubahan pada test set. Evaluasi keluaran probabilitas dilengkapi dengan Brier
score, calibration slope, dan calibration-in-the-large sebagaimana dijelaskan
pada Subbab 2.2.8. Evaluasi tersebut dilakukan pada model yang menyediakan
probabilitas, sedangkan SVM dengan keluaran margin tidak dihitung metrik
kalibrasinya.

Recall\@5%FPR dihitung dari kurva ROC pada test set dengan memilih titik yang
menghasilkan Recall tertinggi selama FPR tidak melebihi 0,05. Perhitungan tidak
menggunakan interpolasi sehingga nilai yang dilaporkan berasal dari titik operasi
yang tersedia pada skor model. FPR aktual dan ambang yang bersesuaian turut
dicatat karena distribusi skor dapat menyebabkan FPR yang dicapai berada di
bawah target. Ambang ini digunakan untuk merangkum titik operasi pada kurva ROC
test set dan dibedakan dari ambang berbasis validation set untuk pelaporan
F1-score, Precision, dan Recall. Perhitungan dapat menggunakan probabilitas
maupun margin fungsi keputusan. Apabila data evaluasi hanya memuat satu kelas,
Recall\@5%FPR dinyatakan tidak tersedia.

Analisis explainability dirancang untuk mengukur kesepakatan antar client
mengenai kepentingan fitur serta perubahan kesepakatan tersebut pada kondisi IID
dan Non-IID. Pengukuran utama menjelaskan satu model global akhir yang sama
menggunakan dua komponen, yaitu background data sebagai distribusi referensi dan
explanation data sebagai sampel yang dijelaskan. Dengan mempertahankan model dan
explanation data, perbandingan diarahkan untuk mengamati pengaruh perbedaan
background lokal terhadap hasil penjelasan.

Background data diambil sebanyak 100 sampel dari data pelatihan lokal yang
digunakan oleh setiap client, termasuk hasil SMOTE apabila oversampling
diterapkan. Jumlah tersebut digunakan untuk menyeragamkan ukuran referensi dan
membatasi biaya komputasi. Explanation data menggunakan subset tetap sebanyak
500 sampel dari test set terpusat yang identik bagi seluruh client. Pengambilan
sampel mengikuti proporsi kelas pada test set sejauh dimungkinkan oleh ukuran
subset. Jumlah fraud yang benar-benar terpilih turut dicatat karena prevalensi
yang rendah dapat menghasilkan sangat sedikit sampel fraud dalam explanation
data. Pemilihan explainer disesuaikan dengan karakteristik model, sebagaimana
diringkas pada @tab-3-explainer.

#figure(
  kind: table,
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.header([*Model*], [*Explainer*], [*Pengaturan utama*]),
    [LR], [LinearSHAP], [Interventional dengan background lokal],
    [SVM linear], [LinearSHAP],
    [Interventional; menjelaskan margin fungsi keputusan],
    [GBM dan XGBoost terpusat], [TreeSHAP],
    [Interventional dengan background yang sesuai konfigurasi],
    [FFD dan BERT], [KernelSHAP],
    [Background lokal yang diringkas dengan k-means],
    [FedXGBllr], [KernelSHAP],
    [Menjelaskan keluaran gabungan ensemble pohon dan CNN],
  ),
  caption: [Pemetaan explainer per model],
) <tab-3-explainer>

LinearSHAP digunakan pada LR dan SVM linear karena memungkinkan perhitungan
atribusi eksak untuk model linear berdasarkan distribusi referensi yang
ditetapkan. TreeSHAP menggunakan mode interventional agar background lokal
menjadi bagian dari definisi penjelasan. Mode ini sesuai dengan tujuan
pengukuran yang membandingkan penjelasan model global dari distribusi referensi
client yang berbeda. KernelSHAP digunakan pada FedXGBllr karena model akhirnya
mencakup gabungan ensemble pohon dan jaringan CNN. FFD dan BERT juga menggunakan
KernelSHAP agar keduanya dievaluasi dengan estimator yang sama berdasarkan
pemeriksaan kelayakan awal terhadap explainer berbasis gradien.

Atribusi dihitung pada skala log-odds untuk model yang menyediakan keluaran
probabilitas sedangkan SVM dijelaskan pada skala margin fungsi keputusan. Pada
FFD, BERT, dan FedXGBllr, keluaran yang dijelaskan diambil langsung dari logit
sebelum aktivasi keluaran. Khusus FedXGBllr, penggunaan aktivasi pra-Sigmoid
menghindari hilangnya variasi skor akibat pemotongan probabilitas sebelum
transformasi logit. Karena skala keluaran SVM berbeda, perbandingan lintas model
berfokus pada peringkat dan kesepakatan fitur dan bukan pada kesamaan magnitudo
atribusi.

Pada setiap client, nilai SHAP diringkas menjadi vektor feature importance
melalui rerata nilai absolut atribusi pada seluruh explanation data. Untuk fitur
$j$ pada client $c$, perhitungannya mengikuti @eq-gcj. Vektor tersebut kemudian
dihimpun untuk menghasilkan ringkasan kepentingan fitur dan ukuran kesepakatan
antar client.

Terdapat empat ringkasan utama yang digunakan dalam analisis. Pertama, rata-rata
feature importance antar client menunjukkan fitur yang secara umum memberikan
kontribusi terbesar terhadap prediksi. Kedua, rata-rata Spearman rank
correlation pada seluruh pasangan client mengukur kesesuaian urutan kepentingan
fitur. Ketiga, Jaccard similarity pada lima fitur teratas mengukur kesamaan
himpunan fitur yang dianggap paling penting. Keempat, indeks Kuncheva digunakan
untuk mengoreksi kesamaan himpunan fitur terhadap peluang pemilihan secara acak.
Koreksi ini diperlukan karena ketiga dataset memiliki jumlah fitur yang berbeda
sehingga nilai Jaccard\@5 tidak langsung digunakan sebagai dasar perbandingan
lintas dataset.

Analisis tambahan dilakukan dengan menghitung indeks Kuncheva pada berbagai
ukuran himpunan fitur teratas. Profil tersebut digunakan untuk mengamati
perubahan kesepakatan ketika semakin banyak fitur disertakan. Korelasi peringkat
berbobot magnitudo juga dilaporkan untuk mengurangi dominasi fitur dengan
atribusi mendekati nol dalam pengukuran peringkat penuh. Apabila vektor
importance konstan atau seluruhnya nol, korelasi dinyatakan tidak terdefinisi dan
konfigurasi ditandai untuk pemeriksaan lebih lanjut dan bukan diberi nilai
kesepakatan nol atau satu.

Ketidakpastian KernelSHAP diperiksa melalui dua pengulangan seed koalisi pada
setiap client dengan model, background, dan explanation data yang tetap.
Korelasi antara kedua vektor importance dari client yang sama digunakan sebagai
acuan reliabilitas estimator yang dalam analisis disebut noise floor. Pengukuran
ini dilakukan pada setiap konfigurasi agar variasi estimator tidak diwakili oleh
satu nilai acuan yang diterapkan ke seluruh dataset. Perbandingan antar client
menggunakan seed koalisi yang sama pada setiap pengulangan sebagai bentuk common
random numbers.

Untuk membandingkan kesepakatan dalam-client dan antar-client, sepuluh vektor
importance dari lima client dan dua seed dianalisis melalui enumerasi seluruh
945 perfect matching, yaitu seluruh cara membentuk lima pasangan dari sepuluh
vektor. Statistik yang digunakan adalah selisih antara rata-rata korelasi pada
lima pasangan dalam suatu matching dan rata-rata korelasi pada 40 pasangan di
luarnya. Pemasangan asli menghubungkan dua pengulangan dari client yang sama.
Nilai p dihitung dari proporsi matching dengan statistik sekurang-kurangnya
sebesar statistik pemasangan asli. Statistik ini dibedakan dari ringkasan
kesepakatan antar client yang dihitung menggunakan 20 pasangan dengan seed yang
sama.

Interpretasi nilai p tersebut bergantung pada asumsi exchangeability, yaitu
bahwa distribusi bersama vektor tetap sama ketika pemasangannya dipertukarkan di
bawah hipotesis nol. Kesamaan importance yang mendasari seluruh client tidak
dengan sendirinya menjamin asumsi tersebut, terutama apabila variansi estimator
berbeda antar client atau penggunaan seed bersama membentuk ketergantungan
tertentu. Oleh karena itu, enumerasi lengkap diperlakukan sebagai perhitungan
eksak terhadap distribusi matching yang ditetapkan sedangkan validitas
inferensinya tetap bergantung pada kesesuaian asumsi pengujian. Koreksi
Benjamini--Hochberg diterapkan pada kelompok pengujian KernelSHAP multi-client
dengan batas nilai p terkoreksi sebesar 0,05. Hasil yang tidak signifikan tidak
ditafsirkan sebagai bukti bahwa interpretasi antar client identik atau stabil.

Skema pengukuran dua seed diringkas pada @fig-3-two-seeds-gap. Ringkasan setiap
konfigurasi memuat reliabilitas dalam-client, kesepakatan antar-client, selisih
keduanya, nilai p, dan nilai p terkoreksi. Rasio kesepakatan antar-client
terhadap noise floor digunakan sebagai indikator tambahan. Dengan pengaturan
ini, nilai antar-client yang berada di bawah noise floor tidak langsung
dikategorikan sebagai ketidakstabilan tanpa mempertimbangkan hasil pengujian dan
besarnya perbedaan.

#figure(
  image("resources/fig-3-two-seeds-gap.png", width: 82%),
  caption: [Skema pengukuran dua-seed. Panel A: setiap client menghasilkan dua vektor importance pada dua seed koalisi, sehingga kesepakatan dalam-client menjadi floor dan perbandingan antar-client dipasangkan pada seed yang sama (common random numbers). Panel B: sebaran kedua ukuran pada satu sel nyata (BAF BERT Non-IID tanpa SMOTE). Di bawah hipotesis nol kedua sebaran berimpit, sehingga floor tidak dapat dipakai sebagai ambang deteksi dan digantikan oleh uji exchangeability eksak atas seluruh 945 perfect matching.],
) <fig-3-two-seeds-gap>

Hasil evaluasi explainability dilaporkan untuk setiap kombinasi model, skema
partisi, dan konfigurasi SMOTE. Rata-rata importance digunakan untuk menjelaskan
fitur yang dominan sedangkan Spearman, Jaccard, dan Kuncheva digunakan untuk
mengukur kesepakatan interpretasi. Perubahan antara kondisi IID dan Non-IID
ditafsirkan sesuai pengaturan analisisnya sebagai sensitivitas terhadap
background lokal. Penafsiran tersebut juga mempertimbangkan ketidakpastian
estimator agar variasi komputasi tidak langsung dianggap sebagai perbedaan
interpretasi antar client.

== Implementasi Perancangan Sistem

Subbab ini menjelaskan implementasi komponen sistem yang telah dirancang pada
Subbab 3.3, mulai dari pembagian data dan preprocessing, partisi client,
penanganan class imbalance, pelatihan dan agregasi model, hingga evaluasi
performa dan explainability. Urutan pembahasan mengikuti struktur perancangan
agar hubungan antara rancangan dan implementasinya dapat ditelusuri.
Implementasi menggunakan Python dengan pustaka pendukung untuk simulasi
federated learning, pelatihan model, analisis SHAP, dan pengelolaan eksperimen.
Rincian perangkat lunak yang digunakan disajikan pada @tab-3-5.

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
    [Pengujian regresi], [pytest],
    [Version control], [Git + GitHub],
  ),
  caption: [Spesifikasi Lingkungan Pengembangan],
) <tab-3-5>

=== Implemetasi Tahap Preprocessing

Preprocessing diimplementasikan secara terpisah untuk setiap dataset dengan
keluaran berupa matriks fitur numerik, label target, dan nama fitur hasil
transformasi. Data terlebih dahulu dibagi menjadi training set, validation set,
dan test set. Parameter transformasi kemudian dihitung hanya dari training set
dan digunakan kembali pada kedua subset lainnya tanpa dilakukan fitting ulang.
Dengan demikian, statistik dari data validasi dan pengujian tidak digunakan
untuk menentukan parameter preprocessing.

Pada PaySim, tahapan implementasi meliputi penghapusan kolom yang tidak
digunakan, pembentukan fitur `errorBalanceOrig` dan `errorBalanceDest`, one-hot
encoding pada jenis transaksi, serta scaling terhadap seluruh matriks fitur.
Pada ULB Credit Card, StandardScaler hanya diterapkan pada fitur Time dan
Amount, sedangkan V1--V28 dipertahankan sebagaimana tersedia. Pada BAF,
implementasi mencakup penghapusan kolom yang telah ditentukan, penggantian nilai
sentinel menjadi nilai kosong, pembentukan indikator missing, imputasi median,
one-hot encoding fitur kategorikal, dan scaling.

Hasil transformasi digunakan sebagai masukan bersama bagi seluruh model pada
dataset yang sama sehingga dimensi dan urutan fitur tetap konsisten. Nama fitur
turut disimpan untuk menghubungkan atribusi SHAP dengan fitur yang dijelaskan.
Matriks fitur akhir memiliki 13 kolom pada PaySim, 30 kolom pada ULB Credit
Card, dan 55 kolom pada BAF.

=== Implementasi Modul Partisi Client

Modul partisi menerima training set hasil preprocessing dan menghasilkan indeks
sampel untuk lima client. Pada skenario IID, sampel dibagi dengan jumlah yang
relatif merata dan proporsi label yang mendekati distribusi training set. Pada
skenario Non-IID, pembagian dilakukan secara terpisah untuk setiap kelas
menggunakan distribusi Dirichlet. Vektor proporsi yang dihasilkan menentukan
alokasi sampel kelas tersebut ke masing-masing client. Parameter konsentrasi
pada eksperimen pelatihan utama ditetapkan sebesar $alpha$ = 0,5.

Pembentukan partisi menggunakan random seed 42 agar pembagian dapat direplikasi.
Indeks partisi yang sama digunakan oleh seluruh model pada kombinasi dataset dan
skema partisi yang sama. Konfigurasi dengan dan tanpa SMOTE juga menggunakan
partisi awal yang sama sehingga perbandingan keduanya tidak dipengaruhi oleh
perbedaan alokasi sampel ke client.

Partisi hanya diterapkan pada training set sesuai dengan pembagian data pada
Subbab 3.3.1. Validation set dan test set tetap utuh di server simulasi untuk
mendukung evaluasi yang konsisten. Setiap client memperoleh subset pelatihan
lokal $D_k subset D_"train"$ tanpa pembagian tambahan menjadi local validation
set. Pada konfigurasi dengan SMOTE, subset tersebut diteruskan ke modul
oversampling lokal sebelum digunakan untuk pelatihan. Pada konfigurasi tanpa
SMOTE, data lokal digunakan secara langsung.

=== Implementasi Modul SMOTE Lokal

Modul SMOTE lokal diimplementasikan menggunakan pustaka imbalanced-learn sesuai
rancangan pada Subbab 3.3.3. Modul menerima matriks fitur dan label dari partisi
masing-masing client kemudian melakukan oversampling sebelum data digunakan
untuk pelatihan model lokal. Proses pembentukan sampel sintetis dilakukan secara
terpisah pada setiap client tanpa memerlukan penggabungan data mentah antar
client.

Konfigurasi SMOTE menggunakan `sampling_strategy` sebesar 0,01 dan `k_neighbors`
sebesar 5 untuk seluruh dataset. Sebelum menjalankan oversampling, modul
memeriksa jumlah sampel fraud dan rasio fraud terhadap non-fraud pada partisi
lokal. Apabila jumlah sampel fraud kurang dari enam, SMOTE dilewati karena
persyaratan jumlah tetangga tidak terpenuhi. Proses juga dilewati apabila rasio
kelas telah mencapai atau melampaui target 0,01. Pada kedua kondisi tersebut,
modul mengembalikan data asli untuk digunakan dalam pelatihan. Pemeriksaan
jumlah minimum ini memastikan kelayakan eksekusi algoritma tetapi tidak menjamin
bahwa pola kelas minoritas terwakilkan.

Setiap pemanggilan modul mencatat jumlah sampel fraud dan non-fraud sebelum
oversampling, jumlah sampel sintetis yang dihasilkan, serta status penerapan
atau alasan pelewatan SMOTE. Modul juga mencatat multiplier sintesis, yaitu
rasio jumlah sampel sintetis terhadap jumlah sampel fraud asli. Ukuran ini
digunakan untuk menunjukkan besarnya penambahan data sintetis relatif terhadap
sampel minoritas yang tersedia.

Modul dapat dinonaktifkan melalui konfigurasi eksperimen untuk menjalankan
kondisi tanpa SMOTE. Dalam kondisi tersebut, partisi lokal diteruskan langsung
ke proses pelatihan tanpa penambahan sampel. Kedua konfigurasi menggunakan
partisi awal yang sama agar perbandingan performa dapat dikaitkan dengan
penerapan aturan SMOTE. Validation set dan test set tidak diproses oleh modul
ini sehingga distribusi kelas pada data evaluasi tetap dipertahankan.

=== Implementasi Pelatihan model dan Skema Agregasi

Pelatihan federated diimplementasikan menggunakan kerangka kerja Flower. Empat
skema agregasi pada @tab-3-4 direalisasikan melalui strategi yang disesuaikan
dengan karakteristik parameter masing-masing model. Komponen client menjalankan
pelatihan lokal dan mengirimkan hasilnya kepada server sedangkan komponen server
mengatur agregasi, evaluasi model global, dan penghentian pelatihan. Seluruh
konfigurasi pelatihan utama menggunakan random seed 42 untuk proses acak,
seperti pembentukan partisi, pengambilan sampel, dan inisialisasi model. Versi
pustaka dan konfigurasi eksperimen juga dicatat untuk mendukung
reproduksibilitas.

Pada LR dan SVM linear, agregasi FedAvg dilakukan terhadap vektor koefisien dan
bias. LR diimplementasikan menggunakan LogisticRegression dari scikit-learn
dengan solver lbfgs dan batas optimasi `max_iter` sebesar 1.000. SVM linear
menggunakan SGDClassifier dengan loss hinge dan lima epoch pelatihan lokal per
putaran. Setelah pelatihan lokal selesai, parameter dikirimkan ke server dan
dirata-ratakan dengan bobot berdasarkan ukuran data lokal masing-masing client.
Parameter global hasil agregasi kemudian digunakan pada putaran berikutnya.

GBM diimplementasikan menggunakan HistGradientBoostingClassifier untuk mendukung
pelatihan pada dataset berukuran besar. Setiap client melatih model pada data
lokal kemudian dievaluasi menggunakan validation set terpusat. Server memilih
kandidat dengan AUPRC validasi tertinggi sebagai model global. Skema best-model
selection ini mengikuti formulasi pada @eq-bestmodel tanpa melakukan perataan
parameter atau penggabungan struktur pohon antar client.

Selain seleksi antar client, implementasi GBM melakukan seleksi jumlah iterasi
boosting. Setiap model dilatih hingga batas `max_iter` sebesar 100. Prediksi
pada setiap prefix boosting kemudian dievaluasi menggunakan validation set
terpusat. Prefix dengan AUPRC tertinggi dipertahankan sebagai model terpilih,
dengan jumlah iterasi $k^* <= 100$. Seleksi prefix menggunakan validation set
yang sama dengan proses pemilihan model global. Prosedur ini diterapkan pada
konfigurasi dengan dan tanpa SMOTE agar jumlah iterasi dapat menyesuaikan
perkembangan performa validasi pada masing-masing konfigurasi.

FFD diimplementasikan sebagai jaringan 1D CNN sedangkan BERT menggunakan
arsitektur Transformer tabular atau FT-Transformer. Kedua model dilatih secara
lokal menggunakan PyTorch yang kemudian parameter jaringannya diagregasi melalui
skema accuracy-weighted FedAvg dengan metrik performa yang digunakan dalam
pembobotan adalah AUPRC. Bobot setiap client dibentuk dari hasil perkalian
jumlah sampel pelatihan lokal dengan skor AUPRC client kemudian dinormalisasi
terhadap jumlah bobot seluruh client. Apabila SMOTE diterapkan, ukuran data yang
digunakan dalam pembobotan adalah jumlah sampel setelah oversampling.

FedXGBllr diimplementasikan dengan mengacu pada baseline hfedxgboost pada
repositori Flower. Tahap pertama melatih 50 pohon XGBoost pada setiap client
kemudian menghimpun ensemble lokal menjadi ensemble gabungan berukuran
$M times K$. Dengan 50 pohon dan lima client, ensemble gabungan terdiri dari 250
pohon. Pada tahap kedua, keluaran ensemble digunakan oleh jaringan 1D CNN untuk
mempelajari pembobotan kontribusi pohon. Ukuran kernel dan stride ditetapkan
sebesar 50, sesuai jumlah pohon per client. Parameter CNN dilatih secara lokal
dan diagregasi menggunakan FedAvg.

Batas maksimum pelatihan federated ditetapkan sebesar 20 putaran untuk LR, SVM,
GBM, FFD, dan BERT. FedXGBllr mengikuti konfigurasi per dataset, yaitu 20
putaran pada PaySim serta 50 putaran pada ULB Credit Card dan BAF. Batas
tersebut merupakan batas maksimum sedangkan jumlah putaran yang dijalankan
bergantung pada mekanisme penghentian pelatihan. Model akhir dipilih berdasarkan
AUPRC validasi terbaik.

Validation set terpusat digunakan sebagai sumber evaluasi global pada seluruh
skema. Pada LR, SVM, FFD, dan BERT, AUPRC validasi digunakan untuk memantau
perkembangan model global dan menentukan penghentian dini ketika tidak terjadi
peningkatan selama sejumlah putaran berturut-turut. Pada GBM, metrik tersebut
digunakan untuk memilih prefix boosting dan model kandidat terbaik. Pada
FedXGBllr, evaluasi digunakan untuk memantau serta memilih model selama
pelatihan CNN. Penggunaan subset dan metrik validasi yang sama menjaga
konsistensi evaluasi tetapi perbandingan antar model tetap mencakup perbedaan
algoritma, arsitektur, dan aturan agregasi.

Konfigurasi hyperparameter ditetapkan melalui pencarian terbatas dan penggunaan
nilai acuan dari pustaka atau implementasi baseline. Optimasi ekstensif tidak
dilakukan agar ruang lingkup eksperimen tetap berfokus pada perbandingan
konfigurasi pelatihan dan agregasi. Nilai yang digunakan disajikan pada
@tab-3-6.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (1.3fr, 1fr),
      align: (left, left),
      table.header([*Parameter*], [*Nilai*]),
      table.cell(colspan: 2)[_Umum (seluruh model)_],
      [Jumlah client (K)], [5],
      [Global rounds (R)], [20 untuk LR, SVM, GBM, FFD, dan BERT; FedXGBllr 20 pada PaySim serta 50 pada ULB dan BAF, mengikuti baseline hfedxgboost Flower],
      [Dirichlet $alpha$], [0,5 pada pelatihan utama; 0,5 / 1,0 / 5,0 pada sensus diagnostik partisi],
      [Random seed], [42],
      [SMOTE: `k_neighbors`], [5],
      [SMOTE: `sampling_strategy`], [0,01 (1:100), seragam untuk PaySim, ULB, dan BAF],

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

Evaluasi dilaksanakan setelah model akhir ditetapkan berdasarkan performa pada
validation set. Tahap ini mencakup pengukuran performa prediksi dan analisis
explainability. Seluruh model pada dataset yang sama dievaluasi menggunakan test
set dan prosedur perhitungan yang seragam agar hasilnya dapat dibandingkan
secara konsisten.

AUPRC dihitung menggunakan average precision sedangkan F1-score, Precision, dan
Recall dihitung menggunakan ambang yang menghasilkan F1 tertinggi pada
validation set. Recall\@5%FPR dihitung dengan memilih titik pada kurva ROC yang
memberikan Recall tertinggi selama FPR tidak melebihi 5%. FPR aktual juga
dilaporkan karena skor prediksi yang tersedia tidak selalu memungkinkan
pencapaian FPR tepat sebesar 5%. Evaluasi probabilitas dilengkapi dengan Brier
score, calibration slope, dan calibration-in-the-large pada model yang
menyediakan keluaran probabilitas. Recall\@5%FPR ditambahkan setelah pelatihan
utama selesai sehingga pengukurannya dilakukan secara post-hoc. Perhitungan
menggunakan model akhir yang telah disimpan tanpa melakukan pelatihan ulang.

Sebagai analisis pendukung, lingkungan lokal sampel minoritas diperiksa pada
training set sebelum penerapan SMOTE. Setiap sampel minoritas dikelompokkan
berdasarkan jumlah sampel minoritas di antara lima tetangga terdekatnya dengan
mengecualikan sampel itu sendiri. Kategori safe mencakup empat atau lima
tetangga minoritas, borderline dua atau tiga, rare satu, dan outlier tanpa
tetangga minoritas dimana pemeriksaan menggunakan jarak Euclidean pada ruang
fitur hasil preprocessing. Referensi tetangga mencakup training set penuh pada
ULB dan BAF sedangkan PaySim menggunakan subsampel acak sebanyak 500.000 baris
untuk membatasi biaya komputasi. Hasilnya digunakan untuk membantu menjelaskan
karakteristik data dengan tetap mempertimbangkan pengaruh preprocessing dan
cakupan referensi tetangga.

Analisis SHAP dilakukan terhadap model akhir menggunakan background lokal dan
explanation data bersama. Background setiap client terdiri dari 100 sampel yang
diambil tanpa pengembalian dari data pelatihan lokal setelah penanganan class
imbalance. Pada KernelSHAP, sampel tersebut diringkas menjadi 10 sentroid
melalui k-means. Explanation data terdiri dari 500 sampel test set yang sama
bagi seluruh client dan konfigurasi pada dataset yang sama. Dengan
mempertahankan model global dan sampel yang dijelaskan, pengukuran utama
diarahkan untuk mengamati perubahan penjelasan akibat perbedaan distribusi
referensi lokal. Adapun pemilihan explanation data mengikuti proporsi kelas
dengan ketentuan sedikitnya satu sampel fraud.

LinearSHAP digunakan untuk LR dan SVM linear, sedangkan TreeSHAP dengan mode
interventional digunakan untuk GBM dan XGBoost. Model yang dijelaskan merupakan
model akhir yang sama dengan model pada evaluasi performa. FFD, BERT, dan
FedXGBllr menggunakan KernelSHAP dengan anggaran 500 koalisi serta tanpa seleksi
fitur otomatis. Pada FedXGBllr, ensemble pohon dan jaringan CNN dijelaskan
sebagai satu fungsi karena hubungan nonlinier pada kepala CNN tidak memungkinkan
atribusi model gabungan diperoleh hanya melalui penjumlahan berbobot atribusi
setiap pohon.

Setiap client menghasilkan vektor feature importance melalui rata-rata nilai
absolut SHAP pada seluruh explanation data. Pada KernelSHAP, perhitungan diulang
menggunakan dua seed koalisi dengan model, background, dan explanation data yang
tetap. Kesepakatan antara kedua pengulangan dalam client digunakan sebagai acuan
reliabilitas estimator. Sementara itu, perbandingan antar client dilakukan pada
pengulangan dengan seed yang sama untuk mengendalikan variasi akibat sampling
koalisi.

Vektor importance selanjutnya dianalisis menggunakan rata-rata importance antar
client, Spearman rank correlation, Jaccard\@5, dan indeks Kuncheva. Profil
Kuncheva pada berbagai jumlah fitur teratas digunakan untuk memeriksa apakah
kesepakatan berubah ketika cakupan fitur diperluas. Korelasi peringkat berbobot
magnitudo turut dihitung dengan memberikan bobot lebih besar pada fitur yang
memiliki rerata atribusi absolut lebih tinggi. Seluruh ukuran tersebut digunakan
secara bersama untuk membedakan kesepakatan mengenai fitur dominan dari
kesesuaian urutan fitur secara keseluruhan.

Perbandingan kesepakatan dalam-client dan antar-client dilakukan melalui
enumerasi seluruh 945 pemasangan dari sepuluh vektor importance yang dihasilkan
oleh lima client dan dua seed. Nilai p kemudian disesuaikan menggunakan koreksi
Benjamini--Hochberg sebagaimana dirancang pada Subbab 3.3.5. Interpretasi
pengujian tetap memperhatikan asumsi exchangeability dan ketergantungan antar
vektor. Hasil yang tidak signifikan tidak diperlakukan sebagai bukti bahwa
penjelasan antar client identik.

= HASIL DAN PEMBAHASAN

Penelitian dilakukan sebanyak 108 sel yang terdiri atas 3 dataset, 6 model, 3 kondisi
partisi, dan 2 kondisi penanganan class imbalance. Dari 108 sel tersebut, 12 di
antaranya tidak dijalankan sehingga menyisakan 96 sel yang dieksekusi. Sel-sel yang
tidak dijalankan merupakan sel dengan penanganan class imbalance pada dataset BAF
dalam kondisi centralized dan partisi IID. Hal ini disebabkan oleh tingkat fraud
dataset yang lebih tinggi (1,1%) daripada syarat yang ditetapkan penelitian ini untuk
dilakukan penanganan class imbalance (1%). Adapun salah satu metrik penilaian yang
digunakan dalam penelitian ini, yaitu AUPRC dapat memiliki makna berbeda antar
dataset karena masing-masing memiliki batas bawah yang berbeda sesuai dengan tingkat
fraud awal dataset, yakni 0,00129 pada PaySim, 0,00173 pada ULB, dan 0,01103 pada
BAF @saito2015.

== Perbandingan Performa Antar Paradigma Agregasi

=== Performa AUPRC dan Recall\@5%FPR antar dataset dan kondisi

Perbandingan performa antar paradigma agregasi dinilai menggunakan dua metrik
penilaian yaitu AUPRC yang menilai kinerja model secara keseluruhan serta
Recall\@5%FPR yang mengukur persentase kasus fraud yang berhasil dideteksi ketika
tingkat false positive rate dibatasi maksimal 5% atau dengan kata lain yang menilai
kemampuan model mendeteksi fraud pada satu kondisi operasional tertentu.

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha$ = 0,5*],
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
        table.cell(colspan: 2)[*Dirichlet $alpha$ = 0,5*],
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

@tab-4-auprc-ulb, @tab-4-auprc-baf, dan @tab-4-auprc-paysim menyajikan hasil AUPRC
pada dataset ULB, BAF, dan PaySim secara berurutan. Sebagaimana dijelaskan
sebelumnya, nilai-nilai antar dataset tidak dapat langsung dibandingkan karena
AUPRC pada dataset berbeda memiliki batas bawah berbeda. Namun, secara umum,
apabila diambil nilai terbaik dari masing-masing dataset dan dibandingkan dengan
batas bawahnya diperoleh dataset dengan hasil terbaik, yaitu PaySim dengan nilai
terbaik 773 kali batas bawah, ULB dengan nilai terbaik 484 kali batas bawah, dan
BAF dengan nilai terbaik 15 kali batas bawah.

Dari hasil tersebut dapat dilihat bahwa pengaruh federated terhadap performa tidak
mutlak melainkan utamanya bergantung pada model yang digunakan serta dataset dan
juga kondisi partisi federated. Pada ketiga dataset, performa model dengan dasar
tree (GBM dan FedXGBllr) bertahan dari kondisi centralized ke kondisi federated
dengan performa yang masih cukup baik dibandingkan model-model lainnya bergantung
pada dataset. Sebaliknya, performa model berbasis Deep Learning cenderung menurun
dari kondisi centralized ke kondisi federated terutama pada partisi Non-IID. Maka
dari itu, efek federated terhadap performa dapat dipahami bergantung pada interaksi
beberapa variabel sekaligus, yakni model dan dataset yang digunakan serta kondisi
partisinya.

#figure(
  kind: table,
  text(size: 8pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[*Model*],
        table.cell(colspan: 2)[*Centralized*],
        table.cell(colspan: 2)[*Dirichlet $alpha$ = 0,5*],
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
        table.cell(colspan: 2)[*Dirichlet $alpha$ = 0,5*],
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

Adapun @tab-4-rfpr-ulb, @tab-4-rfpr-baf, dan @tab-4-rfpr-paysim menyajikan hasil
Recall\@5%FPR pada dataset ULB, BAF, dan PaySim secara berurutan. Apabila dilihat
antar dataset, kesenjangan performa antara dataset BAF dengan dataset ULB dan
PaySim yang sangat besar pada metrik AUPRC menjadi berkurang pada metrik
Recall\@5%FPR dimana pada dataset BAF nilai yang diperoleh secara rata-rata masih
sekitar 0,5 pada metrik Recall\@5%FPR menunjukkan kelayakan yang masih memadai
dalam hal fraud detection.

Kedua metrik baik AUPRC dan Recall\@5%FPR secara umum menunjukkan karakteristik
yang sama dalam menilai performa masing-masing sel. Perbedaan antar kedua metrik
ini terjadi dimana nilai AUPRC yang buruk tetapi kemampuan deteksi masih memadai
yang ditandai oleh nilai Recall\@5%FPR yang cukup baik. Hal ini dapat terjadi
karena kombinasi antara model dan dataset dapat menghasilkan nilai keseluruhan yang
baik namun buruk pada wilayah presisi tinggi atau sebaliknya. Oleh karena perbedaan
ini maka kedua metrik digunakan secara berdampingan.

=== Performa antar Paradigma Agregasi

Paradigma agregasi tree ensemble pada model FedXGBllr menunjukkan performa yang
lebih baik daripada performa paradigma agregasi FedAvg pada model parametrik LR dan
SVM. Performa terbaik AUPRC LR dan SVM pada PaySim diraih pada nilai 0,656 dan
0,632 dalam kondisi IID dengan SMOTE sedangkan performa terbaik FedXGBllr pada
PaySim diraih pada nilai 0,996 dalam seluruh kondisi baik IID maupun Non-IID.
Adapun pada ULB, performa terbaik LR dan SVM diraih pada nilai 0,772 dan 0,743
sedangkan performa terbaik FedXGBllr diraih pada nilai 0,806. FedXGBllr secara
keseluruhan memiliki performa AUPRC lebih baik daripada performa paradigma agregasi
FedAvg.

Paradigma agregasi Accuracy-weighted FedAvg pada model Deep Learning, yaitu FFD dan
BERT apabila dibandingkan dengan model FedXGBllr memiliki karakteristik yang
berbeda. Model DL seperti FFD dan BERT menunjukkan performa AUPRC yang lebih baik
daripada model FedXGBllr pada kondisi partisi data IID. Hal ini ditunjukkan oleh
perolehan nilai 0,158 dan 0,167 oleh model FFD dan BERT dibandingkan nilai 0,141
oleh model FedXGBllr pada dataset BAF kondisi partisi IID. Berbeda halnya pada
kondisi partisi data Non-IID, penurunan performa yang cukup signifikan dialami oleh
model Deep Learning dengan nilai 0,045 dan 0,045 pada model FFD dan BERT
dibandingkan nilai 0,137 pada model FedXGBllr pada dataset BAF dengan kondisi
partisi Non-IID dengan SMOTE. Hal yang serupa terjadi pada dataset PaySim dengan
kondisi partisi Non-IID tanpa SMOTE dimana model DL memperoleh nilai 0,647 dan
0,660 pada model FFD dan BERT dibandingkan nilai 0,996 pada model FedXGBllr. Hasil
ini menunjukkan model DL yang pada kondisi tertentu dapat menghasilkan performa
lebih baik daripada model FedXGBllr namun juga memiliki stabilitas yang buruk di
kondisi lain terutama pada partisi Non-IID dimana FedXGBllr masih dapat
mempertahankan stabilitasnya.

Secara keseluruhan, performa antar paradigma agregasi bergantung pada karakteristik
dataset dan distribusi data serta metrik penilaian yang digunakan. FedXGBllr dengan
paradigma agregasi tree ensemble memberikan performa yang lebih baik dibandingkan
dengan model LR dan SVM dengan paradigma agregasi FedAvg namun masih sedikit di
bawah performa terbaik model FFD dan BERT dengan paradigma agregasi
Accuracy-weighted FedAvg. Namun di lain sisi, model FFD dan BERT dengan paradigma
agregasi Accuracy-weighted FedAvg memiliki ketahanan stabilitas yang buruk terhadap
kondisi partisi data yang bersifat Non-IID dibandingkan dengan model FedXGBllr yang
masih dapat mempertahankan performanya di bawah kondisi data Non-IID.

=== Analisis Performa pada Dataset BAF

Berdasarkan hasil penelitian, performa AUPRC pada dataset BAF berada di bawah
performa AUPRC pada kedua dataset lainnya, yaitu PaySim dan ULB dimana nilai
terbaik AUPRC pada dataset BAF adalah 0,169 dibandingkan dengan nilai terbaik AUPRC
PaySim dan ULB, yaitu 0,997 dan 0,838. Apabila mempertimbangkan batas bawah
tiap-tiap dataset, maka diperoleh BAF dengan nilai terbaik 15 kali batas bawah
dibandingkan dengan PaySim dengan nilai terbaik 773 kali batas bawah dan ULB dengan
nilai terbaik 484 kali batas bawah. #cite(<dong2026fcorr>, form: "prose") memperoleh
hasil AUPRC pada dataset BAF yang sama disajikan pada @tab-4-baf-auprc-bench

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
      [BERT (studi ini)], [*0,1670*], [federated, Dirichlet $alpha$ = 0,5],
      [GBM (studi ini)], [0,1620], [federated, Dirichlet $alpha$ = 0,5],
      [FFD (studi ini)], [0,1581], [federated, IID],
      [FedXGBllr (studi ini)], [0,1515], [federated, Dirichlet $alpha$ = 0,5],
      [LR (studi ini)], [0,1440], [federated, IID],
      [SVM (studi ini)], [0,1194], [federated, Dirichlet $alpha$ = 0,5],
      [XGBoost (studi ini)], [0,1569], [n/a (terpusat)],
    )
  ],
  caption: [Perbandingan AUPRC test pada BAF Base],
) <tab-4-baf-auprc-bench>

Nilai AUPRC pada dataset BAF terbaik penelitian ini dihasilkan oleh model BERT
yaitu 0,169 pada kondisi centralized dan 0,167 pada kondisi federated non-IID
maupun IID. Adapun pada dataset yang sama, #cite(<dong2026fcorr>, form: "prose")
memperoleh nilai AUPRC 0,1607 dengan model FT-Transformer pada kondisi
centralized. Hal ini
menunjukkan bahwa performa pada kondisi federated belum tentu lebih buruk daripada
performa pada kondisi centralized dan juga performa federated tidak hanya
bergantung pada kondisi partisi. Selain itu, perbandingan ini menunjukkan bahwa
performa pada dataset BAF yang lebih buruk daripada dataset lainnya memang
merupakan tantangan dari dataset BAF itu sendiri.

Hasil ini sekilas tampak bertolak belakang dengan fakta bahwa dataset BAF memiliki
fraud rate yang paling tinggi diantara dataset lainnya dalam penelitian ini. Nilai
fraud rate dataset BAF adalah 1,1% dibandingkan nilai fraud rate dataset ULB
sebesar 0,172% yang berarti nilai fraud rate BAF adalah 6,4 kali lebih besar
daripada ULB. Namun hasil performa AUPRC yang diperoleh menunjukkan dataset BAF
mencapai 15 kali batas bawah dibandingkan ULB yang mencapai 484 kali batas bawah.
Analisis menunjukkan bahwa perbedaan ini disebabkan oleh separabilitas kelas dimana
meskipun kelas minoritas BAF banyak namun mayoritas merupakan outlier sehingga
tidak membentuk pola yang bermakna. Analisis ini didukung oleh bukti berupa
tipologi contoh minoritas sebagaimana pada @fig-typology dan @tab-typology @napierala2016types.

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

== Pengaruh Heterogenitas Distribusi Data dan Penanganan Class Imbalance

=== Pengaruh Heterogenitas Distribusi Data tanpa Oversampling

Analisis pengaruh heterogenitas distribusi data dilakukan pada penelitian tanpa
oversampling terlebih dahulu agar dampak heterogenitas dapat ditangkap secara
terisolasi. Pada hal ini, perbandingan dilakukan pada @tab-4-auprc-ulb, @tab-4-auprc-baf,
dan @tab-4-auprc-paysim kolom IID none dan Non-IID none.
Perbandingan ini menunjukkan bahwa heterogenitas distribusi data tidak secara
langsung menurunkan performa bahkan dalam beberapa kondisi justru sebaliknya.

Pengaruh heterogenitas distribusi data terhadap performa AUPRC pada model
parametrik LR dan SVM serta GBM sama terhadap ketiga dataset dimana performa AUPRC
LR menurun pada kondisi Non-IID dan sebaliknya performa AUPRC SVM dan GBM meningkat
pada kondisi Non-IID. Adapun model FedXGBllr mengalami peningkatan performa AUPRC
dalam kondisi Non-IID pada dataset ULB dan BAF serta tidak mengalami perubahan pada
dataset PaySim. Adapun model berbasis Deep Learning cenderung mengalami penurunan
performa AUPRC dalam kondisi Non-IID kecuali FFD pada dataset ULB.

Pengaruh paling signifikan terhadap performa AUPRC ditunjukkan pada dataset PaySim
dimana peningkatan performa model SVM dari 0,311 dalam kondisi IID menjadi 0,577
dalam kondisi Non-IID. Sebaliknya, penurunan performa model BERT dialami pada
dataset PaySim dari 0,858 dalam kondisi IID menjadi 0,660 dalam kondisi Non-IID.
Maka dari itu, pengaruh heterogenitas distribusi data bergantung pada variabel lain
dan tidak hanya dirinya sendiri. Arah dari pengaruh heterogenitas distribusi data
dipengaruhi oleh model dan agregasi yang digunakan serta pola distribusi data tiap
client.

=== Pengaruh Penanganan Class Imbalance dengan SMOTE

Pengaruh SMOTE pada kondisi partisi data IID secara keseluruhan memberikan efek
positif terhadap performa AUPRC pada kedua dataset ULB dan PaySim dengan seluruh
model terkecuali dengan model SVM pada dataset ULB yang mengalami penurunan
performa sekitar 1,0% setelah SMOTE. Peningkatan performa AUPRC paling signifikan
sebesar 103,2% setelah SMOTE dialami pada dataset PaySim dengan model SVM. Tidak
ada model atau agregasi tertentu yang memperoleh peningkatan yang lebih signifikan.
Besar pengaruh SMOTE dalam kondisi IID bergantung pada performa awal sebelum SMOTE
dan bukan bergantung pada model atau agregasi. Hasil ini menunjukkan bahwa dalam
kondisi IID dimana distribusi data tiap client homogen, penanganan class imbalance
dengan SMOTE umumnya memberikan pengaruh positif. @tab-4-iid-smote menunjukkan efek
SMOTE pada kondisi IID untuk dataset ULB dan PaySim.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto),
      align: (left, right, right, right),
      table.header([*Model*], [*IID none*], [*IID SMOTE*],
        [*$Delta$ SMOTE (IID)*]),
      table.cell(colspan: 4, align: left)[*ULB*],
      [LR], [0,758], [0,768], [+1,3%],
      [SVM], [0,741], [0,734], [−1,0%],
      [GBM], [0,698], [0,811], [+16,3%],
      [FFD], [0,802], [0,809], [+0,8%],
      [BERT], [0,818], [0,823], [+0,6%],
      [FedXGBllr], [0,712], [0,806], [+13,1%],
      table.cell(colspan: 4, align: left)[*PaySim*],
      [LR], [0,612], [0,656], [+7,1%],
      [SVM], [0,311], [0,632], [+103,2%],
      [GBM], [0,995], [0,996], [+0,1%],
      [FFD], [0,756], [0,839], [+11,0%],
      [BERT], [0,858], [0,916], [+6,8%],
      [FedXGBllr], [0,996], [0,996], [+0,0%],
    )
  ],
  caption: [Efek SMOTE terhadap AUPRC pada kondisi IID untuk dataset ULB dan PaySim],
) <tab-4-iid-smote>

Pada kondisi data Non-IID, pengaruh SMOTE beragam untuk dataset PaySim dan ULB
dimana SMOTE cenderung memberikan efek positif untuk dataset ULB dan sebaliknya
cenderung memberikan efek negatif untuk dataset PaySim. Adapun SMOTE secara
menyeluruh memberikan efek negatif untuk dataset BAF. Terkecuali untuk model GBM
yang tidak terpengaruh oleh efek SMOTE, model lainnya mengalami penurunan performa
akibat SMOTE. Efek negatif SMOTE untuk dataset BAF-pun beragam namun efeknya dapat
dikelompokkan per paradigma agregasi. @tab-4-baf-smote menyajikan efek SMOTE pada
kondisi Non IID untuk dataset BAF dikelompokkan menurut paradigma agregasi.

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
  caption: [Efek SMOTE terhadap AUPRC pada kondisi Non-IID untuk dataset BAF],
) <tab-4-baf-smote>

Model GBM dengan agregasi Best-model selection tidak mengalami penurunan performa
setelah SMOTE. Adapun agregasi tree ensemble mengalami penurunan 9,3% yang
merupakan penurunan terkecil diantara agregasi lainnya. Model-model parametrik LR
dan SVM dengan agregasi FedAvg masing-masing mengalami penurunan 30,5% dan 31,9%.
Adapun model Deep Learning FFD dan BERT dengan agregasi accuracy-weighted FedAvg
mengalami penurunan paling signifikan yaitu 71,0% dan 73,2%. Urutan efek negatif
SMOTE terhadap kelompok agregasi ini sama apabila diukur dengan metrik
Recall\@5%FPR sebagaimana pada @tab-4-baf-smote-rfpr. Hal ini menunjukkan bahwa
efek SMOTE juga bergantung pada agregasi model Federated Learning.

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
  caption: [Efek SMOTE terhadap Recall\@5%FPR pada kondisi Non-IID untuk dataset BAF],
) <tab-4-baf-smote-rfpr>

=== Analisis Pengaruh Kumulatif Heterogenitas Distribusi Data dan Penanganan Class Imbalance

Pengaruh heterogenitas distribusi data dan penanganan class imbalance tidaklah
berdiri sendiri melainkan pengaruh keduanya bergantung pada interaksi yang terjadi.
Pengaruh heterogenitas distribusi data sendiri tidak selalu menurunkan performa
justru dalam beberapa kondisi sebaliknya. Adapun pengaruh penanganan class
imbalance dengan SMOTE bergantung pada kondisi oversampling yang terjadi. Maka dari
itu, efek yang terjadi pada performa bergantung pada interaksi kedua variabel dan
bukan hanya salah satu saja.

Interaksi antara kedua variabel ini dimulai ketika heterogenitas data terjadi dalam
kondisi Non-IID sehingga suatu client menerima sedikit kelas minoritas atau
imbalance yang ekstrem. Kemudian proses penanganan class imbalance dengan SMOTE
menghasilkan data sintetis dari sedikit data minoritas tersebut. Hal ini kemudian
menghasilkan performa metrik lokal yang sangat baik dan proses agregasi
menggabungkannya dengan model global yang pada akhirnya dampaknya terlihat pada
performa metrik global.

Besarnya dampak tersebut juga kemudian bergantung pada variabel lain, terutama
aturan agregasi model. Hal ini yang menyebabkan perbedaan penurunan performa AUPRC
pada kondisi Non-IID untuk dataset BAF per aturan agregasi. Penurunan terburuk
sekitar 70% terjadi pada model dengan aturan agregasi accuracy-weighted FedAvg
karena client lokal dengan imbalance ekstrem yang telah mengalami oversampling oleh
SMOTE menghasilkan performa lokal yang sangat baik sehingga bobotnya menjadi besar
dalam agregasi ini. Hal ini pada akhirnya bersifat menyesatkan dan menghasilkan
performa global yang buruk. Maka dari itu, dalam federated learning, penanganan
class imbalance tidak terlepas dari distribusi data dan juga aturan agregasi yang
digunakan.

== Interpretabilitas Model <sec-hasil-rq3>

=== Perbandingan Konsistensi Interpretasi Antar Model

Hasil penelitian menunjukkan bahwa model parametrik LR dan SVM cenderung
menghasilkan interpretasi yang lebih konsisten antar client dibandingkan model
lainnya pada kumpulan konfigurasi yang dievaluasi. Namun, konsistensi tersebut
tidak membentuk urutan yang selalu sama pada seluruh dataset. Model berbasis tree
maupun deep learning juga dapat menghasilkan kesamaan interpretasi yang tinggi
tetapi memiliki variasi yang lebih besar pada kondisi tertentu.
@tab-4-rq3-corr menyajikan median dan rentang korelasi peringkat berbobot antar
client pada pengukuran utama.

#figure(
  kind: table,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Model*], [*Median*], [*Minimum*], [*Maksimum*]),
    [LR], [0,9845], [0,9064], [0,9970],
    [SVM], [0,9895], [0,8980], [0,9989],
    [GBM], [0,9774], [0,8014], [0,9960],
    [FFD], [0,9634], [0,7979], [0,9924],
    [BERT], [0,9517], [0,8682], [0,9949],
    [FedXGBllr], [0,9576], [0,6650], [0,9859],
  ),
  caption: [Korelasi peringkat berbobot antar client pada pengukuran utama],
) <tab-4-rq3-corr>

Model LR dan SVM memperoleh median korelasi sebesar 0,9845 dan 0,9895, diikuti
GBM sebesar 0,9774. Adapun FFD, BERT, dan FedXGBllr memperoleh median sebesar
0,9634, 0,9517, dan 0,9576. Nilai tersebut menunjukkan bahwa secara umum urutan
kepentingan fitur masih memiliki kesamaan yang tinggi antar client. Namun, median
yang tinggi belum menggambarkan seluruh kondisi sebagaimana terlihat pada
FedXGBllr yang memiliki rentang dari 0,6650 hingga 0,9859. Maka dari itu,
karakteristik interpretabilitas perlu dilihat bersama dengan variasinya antar
dataset dan kondisi partisi.

Kesamaan lima fitur terpenting menunjukkan pola yang searah pada kelompok LR,
SVM, dan GBM. Rata-rata indeks Kuncheva mencapai 0,9433 pada SVM, 0,9112 pada LR,
dan 0,8219 pada GBM. Meskipun model parametrik memiliki nilai lebih tinggi secara
keseluruhan, perbandingan per dataset menunjukkan bahwa keunggulan tersebut tidak
berlaku pada seluruh kondisi.

#figure(
  kind: table,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Dataset*], [*LR*], [*SVM*], [*GBM*]),
    [BAF], [0,956], [0,927], [0,853],
    [ULB], [1,000], [0,964], [0,742],
    [PaySim], [0,789], [0,935], [0,878],
  ),
  caption: [Rerata indeks Kuncheva lima fitur terpenting per dataset],
) <tab-4-rq3-kuncheva>

Pada ULB, LR menghasilkan indeks Kuncheva sebesar 1,000 dibandingkan GBM sebesar
0,742. Sebaliknya, pada PaySim, GBM memperoleh nilai 0,878 yang lebih tinggi
daripada LR sebesar 0,789. Hasil ini menunjukkan bahwa model linear tidak selalu
menghasilkan kesamaan fitur terpenting yang lebih tinggi daripada model tree.
Nilai 1,000 pada LR di ULB menunjukkan bahwa himpunan lima fitur terpenting sama
antar client tetapi tidak berarti urutan maupun besaran atribusi seluruh fiturnya
identik.

Adapun pengaruh kondisi partisi menunjukkan bahwa interpretasi cenderung lebih
konsisten pada kondisi IID. Pada FFD, BERT, dan FedXGBllr, korelasi peringkat
berbobot menurun pada kondisi Non-IID dalam 11 dari 15 pasangan yang dapat
dibandingkan. Pola tersebut paling konsisten pada PaySim dimana seluruh enam
pasangan menunjukkan penurunan. Namun, arah yang sama tidak ditemukan pada
seluruh model dan dataset sehingga Non-IID tidak dipahami sebagai penyebab
penurunan yang bersifat mutlak.

Secara keseluruhan, model parametrik cenderung memiliki konsistensi feature
importance yang tinggi sedangkan model tree, deep learning, dan tree ensemble
menunjukkan variasi yang lebih bergantung pada dataset dan kondisi partisi.
Kondisi Non-IID cenderung menurunkan konsistensi tersebut tetapi besarnya
pengaruh berbeda antar model. Perbedaan ini tidak dapat dikaitkan hanya dengan
paradigma agregasi karena arsitektur model dan metode penjelasannya juga berbeda.
Selain itu, konsistensi interpretasi merupakan karakteristik yang berbeda dari
performa deteksi sehingga model dengan AUPRC tinggi belum tentu menghasilkan
penjelasan yang sama pada seluruh client.

=== Pengaruh Heterogenitas Distribusi Data dan SMOTE terhadap Interpretasi

Pengaruh heterogenitas distribusi data pada FFD, BERT, dan FedXGBllr menunjukkan
penurunan rata-rata korelasi peringkat berbobot sebesar 0,054 dari kondisi IID ke
Non-IID pada pengukuran utama. Penurunan terjadi pada 11 dari 15 pasangan, dengan
nilai p sebesar 0,0042 berdasarkan uji Wilcoxon berpasangan satu sisi. Hasil ini
menunjukkan kecenderungan penurunan kesamaan interpretasi meskipun arah perubahan
pada setiap konfigurasi tidak selalu sama.

Pengaruh paling konsisten ditemukan pada PaySim, dimana seluruh enam pasangan
menunjukkan korelasi yang lebih rendah pada kondisi Non-IID. Namun, besar
penurunannya berbeda antar model dan kondisi SMOTE sebagaimana disajikan pada
@tab-4-rq3-paysim.

#figure(
  kind: table,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    table.header([*Model*], [*Kondisi SMOTE*], [*IID*], [*Non-IID*]),
    [FedXGBllr], [Tanpa SMOTE], [0,986], [0,822],
    [FedXGBllr], [Dengan SMOTE], [0,948], [0,665],
    [FFD], [Tanpa SMOTE], [0,973], [0,798],
    [FFD], [Dengan SMOTE], [0,931], [0,886],
    [BERT], [Tanpa SMOTE], [0,983], [0,941],
    [BERT], [Dengan SMOTE], [0,972], [0,952],
  ),
  caption: [Korelasi peringkat berbobot antar client pada PaySim menggunakan
  KernelSHAP tersampel],
) <tab-4-rq3-paysim>

Penurunan terbesar terjadi pada FedXGBllr dengan SMOTE, yaitu dari 0,948 pada
kondisi IID menjadi 0,665 pada kondisi Non-IID. Tanpa SMOTE, nilainya juga
menurun dari 0,986 menjadi 0,822. Adapun FFD tanpa SMOTE mengalami penurunan dari
0,973 menjadi 0,798 sedangkan BERT menunjukkan perubahan yang lebih kecil pada
kedua kondisi SMOTE. Hasil ini menunjukkan bahwa ketiga model memiliki
sensitivitas interpretasi yang berbeda terhadap heterogenitas data meskipun
seluruhnya menggunakan metode penjelasan yang sama.

Perubahan tersebut tidak selalu searah dengan perubahan performa deteksi. Pada
PaySim tanpa SMOTE, AUPRC FedXGBllr tetap sekitar 0,996 pada kondisi IID maupun
Non-IID sedangkan kesamaan interpretasinya menurun. Dengan demikian, kemampuan
model mempertahankan performa deteksi tidak menjamin bahwa peringkat fitur yang
dianggap penting akan tetap konsisten antar client. Dalam pengukuran ini,
penurunan konsistensi menunjukkan bahwa penjelasan model global menjadi lebih
sensitif terhadap perbedaan background lokal.

Pengaruh Non-IID juga ditemukan pada model dengan explainer deterministik. Pada
GBM di PaySim tanpa SMOTE, indeks Kuncheva menurun dari 1,000 pada kondisi IID
menjadi 0,7725 pada kondisi Non-IID. Pada ULB tanpa SMOTE, nilainya menurun dari
0,688 menjadi 0,544. Namun, LR di ULB tetap memiliki indeks Kuncheva 1,000 pada
kedua kondisi. Maka dari itu, heterogenitas tidak selalu mengubah himpunan lima
fitur terpenting dan pengaruhnya tetap bergantung pada model serta dataset.

Penanganan class imbalance dengan SMOTE turut memengaruhi konsistensi
interpretasi. Pada kelompok LR, SVM, dan GBM, rata-rata kenaikan indeks Kuncheva
setelah SMOTE sebesar 0,048 pada enam pasangan IID dan 0,085 pada sembilan
pasangan Non-IID. Salah satu peningkatan terbesar terjadi pada GBM di ULB kondisi
Non-IID, yaitu dari 0,544 menjadi 0,904. Adapun LR dan SVM di BAF kondisi Non-IID
meningkat dari 0,868 menjadi 1,000.

#figure(
  kind: table,
  text(size: 9pt)[
    #set par(justify: false)
    #table(
      columns: (auto, auto, auto, 1.5fr),
      align: (left, right, right, left),
      table.header([*Dataset dan model*], [*Kuncheva tanpa SMOTE*],
        [*Kuncheva dengan SMOTE*], [*Perubahan fitur penting*]),
      [ULB -- GBM], [0,544], [0,904],
      [`V7` keluar dari lima fitur terpenting, sedangkan `V3` masuk pada peringkat
      ketiga seluruh client],

      [BAF -- LR], [0,868], [1,000],
      [Fitur dominan berubah dari `prev_address_months_count_missing` menjadi
      `housing_status_BB` pada tiga dari lima client; satu client menempatkan
      `housing_status_BA` pada peringkat pertama dan satu client lainnya tetap
      menempatkan `prev_address_months_count_missing`],

      [BAF -- SVM], [0,868], [1,000],
      [Lima fitur terpenting setelah SMOTE terdiri dari empat indikator
      `housing_status` dan `has_other_cards`],
    )
  ],
  caption: [Perubahan konsistensi dan fitur penting setelah SMOTE pada beberapa
  konfigurasi Non-IID],
) <tab-4-rq3-smote>

Peningkatan konsistensi tersebut disertai perubahan fitur yang dianggap penting.
Pada GBM di ULB, `V7` yang sebelumnya berada pada peringkat kedua seluruh client
tidak lagi termasuk dalam lima fitur terpenting setelah SMOTE. Sebaliknya, `V3`
masuk pada peringkat ketiga seluruh client. Pada LR di BAF, fitur dominan berubah
dari indikator ketidaktersediaan riwayat alamat menjadi kategori status tempat
tinggal pada empat dari lima client, sedangkan satu client lainnya tetap
menempatkan indikator ketidaktersediaan riwayat alamat pada peringkat pertama.
Hasil ini menunjukkan bahwa SMOTE dapat meningkatkan kesamaan interpretasi dengan
membentuk kesepakatan terhadap kelompok fitur yang berbeda.

Namun, pengaruh positif terhadap konsistensi tidak ditemukan pada seluruh model.
Pada PaySim kondisi Non-IID, korelasi FFD meningkat dari 0,798 menjadi 0,886
setelah SMOTE dan BERT meningkat dari 0,941 menjadi 0,952. Sebaliknya, FedXGBllr
menurun dari 0,822 menjadi 0,665. Dengan demikian, pengaruh SMOTE terhadap
interpretasi bergantung pada interaksinya dengan model dan distribusi data
sebagaimana pengaruhnya terhadap performa prediksi.

Peningkatan konsistensi juga belum tentu menunjukkan peningkatan kualitas
deteksi. Pada LR di BAF kondisi Non-IID, indeks Kuncheva meningkat dari 0,868
menjadi 1,000 sedangkan AUPRC menurun dari sekitar 0,139 menjadi 0,097. Maka dari
itu, kesepakatan antar client mengenai fitur terpenting perlu dibaca bersama
dengan isi penjelasan dan performa model. Perubahan setelah SMOTE mencakup model
yang dilatih serta background yang digunakan sehingga belum dapat dikaitkan hanya
dengan salah satu komponen tersebut.

=== Keterandalan dan Batas Interpretasi Hasil SHAP

Konsistensi interpretasi antar client perlu dibedakan dari konsistensi pengukuran
SHAP itu sendiri. LR, SVM, dan GBM menghasilkan atribusi yang sama ketika
perhitungan diulang dengan masukan tetap. Adapun FFD, BERT, dan FedXGBllr
menggunakan KernelSHAP tersampel sehingga perbedaan hasil dapat berasal dari
distribusi data antar client maupun variasi sampling. Pengulangan dua seed pada
client yang sama digunakan untuk mengukur besarnya variasi pengukuran tersebut.

#figure(
  kind: table,
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto),
      align: (left, right, right, right),
      table.header([*Model*], [*Median kesepakatan dua seed dalam client*],
        [*Median kesepakatan antar client*],
        [*Konfigurasi dengan perbedaan terdeteksi*]),
      [FFD], [0,9997], [0,9634], [11 dari 11],
      [BERT], [0,9993], [0,9517], [10 dari 11],
      [FedXGBllr], [0,9988], [0,9576], [8 dari 11],
    )
  ],
  caption: [Kesepakatan pengulangan dan kesepakatan antar client pada KernelSHAP
  tersampel],
) <tab-4-rq3-seeds>

Kesepakatan dua pengulangan dalam client mendekati 1 pada ketiga model sedangkan
kesepakatan antar client lebih rendah. Hasil ini menunjukkan bahwa pengulangan
pada masukan yang sama menghasilkan peringkat yang lebih serupa dibandingkan
pengukuran menggunakan background client yang berbeda. Berdasarkan prosedur
pengujian dan koreksi pengujian berganda yang digunakan, 29 dari 33 konfigurasi
menghasilkan indikasi perbedaan antar client.

Kesamaan peringkat yang tinggi dengan demikian masih dapat disertai perbedaan
yang terdeteksi secara statistik. Sebaliknya, empat konfigurasi yang tidak
menunjukkan perbedaan signifikan belum memberikan bukti yang cukup untuk
membedakan variasi antar client dari variasi estimator. Hasil tersebut tidak
langsung berarti bahwa interpretasinya identik atau stabil.

Kesamaan himpunan fitur, urutan fitur, dan besaran atribusi juga memiliki arti
yang berbeda. Indeks Kuncheva sebesar 1,000 menunjukkan kesamaan himpunan fitur
terpilih tetapi tidak memastikan bahwa urutan maupun kontribusi setiap fitur
sama. Demikian pula, korelasi peringkat yang mendekati 1 menunjukkan urutan yang
hampir seragam tanpa mengharuskan besaran SHAP identik. Oleh karena itu, hasil
konsistensi pada penelitian ini terutama menjawab kesepakatan mengenai fitur
penting dan peringkatnya.

Secara keseluruhan, hasil pengukuran mendukung pembacaan bahwa konsistensi
interpretasi bergantung pada model, dataset, dan kondisi distribusi data. Namun,
tingginya satu nilai kesamaan belum cukup untuk menyatakan kualitas explainability
secara menyeluruh. Penilaian perlu mempertimbangkan fitur yang dianggap penting,
perubahan peringkatnya, serta keterandalan pengukurannya. Temuan ini berlaku pada
model akhir yang dievaluasi dalam penelitian dan belum mengukur kestabilan
interpretasi lintas pengulangan pelatihan.

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
