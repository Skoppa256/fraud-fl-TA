//=============================================================================
// ITS Informatics Department Thesis (Typst)
//
// This is your main entry point. Edit the content below and
// customize the configuration variables at the top of this file
// for your thesis metadata.
//
// Department of Informatics
// Faculty of Intelligent Electrical and Informatics Technology
// Institut Teknologi Sepuluh Nopember (ITS)
// Surabaya, Indonesia
//=============================================================================

//=============================================================================
// THESIS CONFIGURATION
// Edit the variables below to customize your thesis metadata.
// All variables are available directly (without prefix) throughout
// this file and are also passed to the template as named arguments.
//=============================================================================

// --- Author Information ---
#let author = "Amadeo Yesa"
#let nrp = "5025231160"
#let sign = "resources/fake-sign.svg"

// --- Supervisor Information ---
#let supervisors = (
  (name: "Moch Nafkhan Alzamzami, S.T., M.T.", nip: "199911222024061001", sign: "resources/fake-sign.svg"),
  (name: "Muhammad Arif Faizin, S.Kom., M.Kom.", nip: "2001202611009", sign: "resources/fake-sign.svg"),
)

// --- Examiner Information ---
// TODO: isi nama & NIP penguji sidang setelah jadwal sidang ditetapkan.
#let examiners = (
  (name: "Dosen Penguji ke-1", nip: "20xxxxxxxx", sign: "resources/fake-sign.svg"),
  (name: "Dosen Penguji ke-2", nip: "20xxxxxxxx", sign: "resources/fake-sign.svg"),
  (name: "Dosen Penguji ke-3", nip: "20xxxxxxxx", sign: "resources/fake-sign.svg"),
)

// --- Head of Department ---
// TODO: isi nama & NIP Kepala Departemen Teknik Informatika ITS.
#let chief = (name: "Nama Kepala Departemen Informatika", nip: "20xxxxxxxx", sign: "resources/fake-sign.svg")

// --- Dates ---
// TODO: perbarui tanggal penulisan/sidang final saat maju sidang Tugas Akhir.
#let dates = (
  writing: "Mei 2026",
  exam: (day: "TODO", date: "TODO", place: "TODO"), // TODO: detail sidang akhir
  writingPeriod: "Mei 2026",
)

// --- Academic Program ---
#let program = (
  title: "Sarjana Komputer (S.Kom.)",
  degree: "S-1",
  concentration: "Teknik Informatika",
  department: "Departemen Teknik Informatika",
  faculty: "Fakultas Teknologi Elektro dan Informatika Cerdas",
  university: "Institut Teknologi Sepuluh Nopember",
  city: "Surabaya",
  year: 2026,
)

// --- Academic Program (English) ---
#let program-en = (
  title: "Bachelor of Computer Science (B.Comp.Sc.)",
  degree: "S1",
  concentration: "Informatics",
  department: "Department of Informatics",
  faculty: "Faculty of Intelligent Electrical and Informatics Technology",
  university: "Institut Teknologi Sepuluh Nopember",
  city: "Surabaya",
  year: 2026,
)

// --- Essay Code ---
#let essay = (
  id: "Tugas Akhir - EF234702",
  en: "Final Project - EF234702"
)

// --- Thesis Titles ---
// Judul tugas akhir ditulis singkat, jelas, dan menggambarkan tema pokok
#let title = (
  id: "Evaluasi Performa dan Analisis Explainability pada Federated Learning Berbasis Tree (FedXGBllr) untuk Financial Fraud Detection",
  en: "Performance and Explainability Evaluation of Tree-Based Federated Learning (FedXGBllr) for Financial Fraud Detection",
)

// --- Resource Paths (relative to src/) ---
#let paths = (
  logo: "resources/its-logo.png",
  coverBackground: "resources/its-thesis-cover-without-logo.svg",
  coverBackgroundSecondary: "resources/its-thesis-cover-without-logo-2.svg",
  bibliography: "bibliography.bib",
)

// HELPER
#let tab-to(target-width, body) = context {
  let current-width = measure(body).width
  if current-width < target-width {
    h(target-width - current-width)
  } else {
    h(0pt)
  }
}

//=============================================================================
// TEMPLATE SETUP
// Each config variable is passed directly as a named argument to the
// template — no `cfg` wrapper needed. Use them directly in your content.
//=============================================================================

#import "template.typ": thesis

#show: thesis.with(
  author: author,
  nrp: nrp,
  sign: sign,
  supervisors: supervisors,
  examiners: examiners,
  chief: chief,
  dates: dates,
  program: program,
  program-en: program-en,
  essay: essay,
  title: title,
  paths: paths,
  proposal: false, // true or false
)

// Indonesian citation formatting replace "et al." with "dkk"
#show "et al.": "dkk"
// Replace " & " with " dan " for Indonesian convention
#show " & ": " dan "

//=============================================================================
// 1. HALAMAN PERSEMBAHAN
//=============================================================================

#heading("HALAMAN PERSEMBAHAN", numbering: none)

// Body intentionally left empty — title only, per the author (2026-10-05).

#pagebreak()

//=============================================================================
// 2. ABSTRAK
//=============================================================================

#heading("ABSTRAK", numbering: none)

// Typst 0.13+ doesn't support par(indent: ...)
// For abstracts, we can leave paragraphs without indent or use #par.leading

#v(1em)
#align(center, text(size: 12pt, weight: "bold")[
  #upper(title.id)
])

#v(1em)
#par(first-line-indent: 0pt)[
  Nama Mahasiswa / NRP #tab-to(4.4cm, [Nama Mahasiswa / NRP]): #author / #nrp \
  Departemen #tab-to(4.4cm, [Departemen]): #program.department \
  Dosen Pembimbing #tab-to(4.4cm, [Dosen Pembimbing]): #supervisors.at(0).name \
  Dosen Ko-pembimbing #tab-to(4.4cm, [Dosen Ko-pembimbing]): #supervisors.at(1).name
]

#v(2em)
#block(align(left, text(size: 13pt, weight: "bold")[Abstrak]))

#v(1em)
Digitalisasi sektor keuangan tidak hanya meningkatkan risiko financial fraud namun juga kompleksitasnya. Walaupun financial fraud detection telah berkembang dengan menggunakan centralized machine learning, metode ini memiliki keterbatasan privasi yang merupakan aspek penting dalam sektor keuangan. Federated Learning (FL) menyediakan alternatif untuk kolaborasi antar institusi tanpa adanya perpindahan data mentah. Namun, eksplorasi model berbasis tree dalam FL untuk financial fraud detection masih terbatas dan juga penelitian yang membandingkan berbagai model di bawah kondisi Non-IID dan class imbalance ekstrem beserta analisis explainability-nya masih minim. Penelitian ini mengevaluasi performa serta menganalisis explainability dari enam model FL dengan empat paradigma agregasi yang berbeda, yaitu Logistic Regression (LR) dan Support Vector Machine (SVM) dengan FedAvg, Gradient Boosting Machine (GBM) dengan best-model selection, FFD dan BERT dengan accuracy-weighted FedAvg, serta FedXGBllr dengan tree ensemble aggregation. Penelitian dilakukan menggunakan tiga dataset, yaitu PaySim, ULB Credit Card, dan Bank Account Fraud (BAF) pada simulasi cross-silo FL berbasis kerangka kerja Flower dengan partisi Dirichlet untuk simulasi kondisi Non-IID dan penerapan SMOTE lokal untuk menangani class imbalance. Evaluasi menggunakan AUPRC sebagai metrik utama sementara analisis explainability dilakukan menggunakan SHAP values per-client untuk mengukur stabilitas feature importance. Hasil penelitian menunjukkan bahwa selisih terbesar AUPRC antara model FL terbaik dan model terpusat terbaik sebesar 1,4%, dengan AUPRC model FL tertinggi 0,996. Meskipun demikian, performa sangat bervariasi antar model, dengan AUPRC terendah 0,311 pada dataset yang sama dengan AUPRC tertinggi. Heterogenitas data tidak selalu menurunkan performa karena ditemukan bahwa model berbasis tree relatif stabil, sedangkan model deep learning dapat mengalami penurunan AUPRC hingga 30%. Selain itu, ditemukan bahwa penanganan class imbalance dengan SMOTE pada kondisi Non-IID menurunkan performa AUPRC ketika partisi Non-IID menyisakan sangat sedikit sampel fraud pada beberapa client. Lebih dari itu, dampak penurunan performa paling signifikan disebabkan oleh interaksi antara suatu paradigma agregasi, kondisi heterogenitas data, dan juga penanganan class imbalance. Hasil penelitian juga menunjukkan konsistensi feature importance berbasis SHAP bergantung pada interaksi antara model, dataset, dan kondisi distribusi data namun model LR dan SVM cenderung menghasilkan konsistensi antar client yang lebih tinggi. Penelitian menunjukkan bahwa hubungan antara performa dan explainability tidak bersifat searah. Ketahanan performa deteksi tidak menjamin konsistensi interpretasi antar client, sedangkan kesamaan interpretasi yang tinggi belum menunjukkan kualitas deteksi yang lebih baik.

#v(1em)
#par(first-line-indent: 0pt)[
  #text(weight: "bold")[Kata Kunci:] Federated Learning, Financial Fraud Detection, FedXGBllr, Explainable AI, SHAP, Non-IID.
]

#pagebreak()

//=============================================================================
// 3. ABSTRACT
//=============================================================================

#heading("ABSTRACT", numbering: none)

#v(1em)
#align(center, text(size: 12pt, weight: "bold")[
  #upper(title.en)
])

#v(1em)
#par(first-line-indent: 0pt)[
  Student Name / NRP #tab-to(4.3cm, [Student Name / NRP]): #author / #nrp \
  Department #tab-to(4.3cm, [Department]): #program-en.department \
  Advisor #tab-to(4.3cm, [Advisor]): #supervisors.at(0).name \
  Co-Advisor #tab-to(4.3cm, [Co-Advisor]): #supervisors.at(1).name
]

#v(2em)
#block(align(left, text(size: 13pt, weight: "bold")[Abstract]))

#v(1em)
The digitalization of the financial sector has increased not only the risk of financial fraud but also its complexity. Although financial fraud detection has advanced through centralized machine learning, this approach is constrained by privacy considerations that are critical in the financial sector. Federated Learning (FL) offers an alternative for cross-institutional collaboration without any movement of raw data. However, the exploration of tree-based models in FL for financial fraud detection remains limited, and studies comparing a range of models under Non-IID conditions and extreme class imbalance, together with their explainability analysis, are still scarce. This study evaluates the performance and analyzes the explainability of six FL models built upon four different aggregation paradigms, namely Logistic Regression (LR) and Support Vector Machine (SVM) with FedAvg, Gradient Boosting Machine (GBM) with best-model selection, FFD and BERT with accuracy-weighted FedAvg, and FedXGBllr with tree ensemble aggregation. The experiments are conducted on three datasets, namely PaySim, ULB Credit Card, and Bank Account Fraud (BAF), in a cross-silo FL simulation on the Flower framework, with Dirichlet partitioning to simulate Non-IID conditions and local application of SMOTE to address class imbalance. Evaluation uses AUPRC as the primary metric, while the explainability analysis uses per-client SHAP values to measure feature importance stability. The results show that the largest AUPRC gap between the best FL model and the best centralized model is 1.4%, with a highest FL AUPRC of 0.996. Nevertheless, performance varies widely across models, with a lowest AUPRC of 0.311 on the same dataset as the highest AUPRC. Data heterogeneity does not always reduce performance, as tree-based models were found to be relatively stable, whereas deep learning models can experience an AUPRC decrease of up to 30%. In addition, handling class imbalance with SMOTE under Non-IID conditions was found to reduce AUPRC when the Non-IID partition leaves very few fraud samples at some clients. Beyond that, the most significant performance degradation stems from the interaction between a given aggregation paradigm, the data heterogeneity condition, and the class imbalance handling. The results also show that the consistency of SHAP-based feature importance depends on the interaction between the model, the dataset, and the data distribution condition, but the LR and SVM models tend to produce higher cross-client consistency. The study shows that performance and explainability do not necessarily move in the same direction. Robust detection performance does not guarantee consistent interpretation across clients, while high interpretation similarity does not by itself indicate better detection quality.

#v(1em)
#par(first-line-indent: 0pt)[
  #text(weight: "bold")[Keywords:] Federated Learning, Financial Fraud Detection, FedXGBllr, Explainable AI, SHAP, Non-IID.
]

#pagebreak()

//=============================================================================
// 4. KATA PENGANTAR
//=============================================================================

#heading("KATA PENGANTAR", numbering: none)

// Body intentionally left empty — title only, per the author (2026-10-05).

#pagebreak()

//=============================================================================
// 5. DAFTAR ISI
//=============================================================================

#heading("DAFTAR ISI", numbering: none)

#outline(
  title: none,
  indent: auto,
  depth: 3,
)

#pagebreak()

//=============================================================================
// 6. DAFTAR TABEL
//=============================================================================

#heading("DAFTAR TABEL", numbering: none)

#outline(
  title: none,
  target: figure.where(kind: table),
)

#pagebreak()

//=============================================================================
// 7. DAFTAR GAMBAR
//=============================================================================

#heading("DAFTAR GAMBAR", numbering: none)

#outline(
  title: none,
  target: figure.where(kind: image),
)

#pagebreak()

#include "content.typ"

// Ensure back matter uses Arabic page numbering
#set page(numbering: "1")

//=============================================================================
// DAFTAR PUSTAKA
//=============================================================================

#heading("DAFTAR PUSTAKA", numbering: none)

#bibliography(
  "bibliography.bib",
  title: none,
  style: "apa",
  full: false,
)

#pagebreak()

//=============================================================================
// LAMPIRAN
//=============================================================================

#heading(upper("Lampiran A. Konfigurasi Eksperimen"), numbering: none)

// Body intentionally left empty — title only, per the author (2026-10-05).

#pagebreak()

//=============================================================================
// BIOGRAFI PENULIS
//=============================================================================

#heading("BIOGRAFI PENULIS", numbering: none)

// Body intentionally left empty — title only, per the author (2026-10-05).
