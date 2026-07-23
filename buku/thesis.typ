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

// TODO: tulis halaman persembahan (tidak ada di Proposal TA v2.1; diisi saat penyusunan buku Tugas Akhir).
#v(1em)
#align(left)[
  Dengan penuh rasa syukur atas selesainya Tugas Akhir ini, penulis mempersembahkan karya ini kepada kedua orang tua, keluarga, serta seluruh pihak yang telah memberikan dukungan dan doa.
]

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
Digitalisasi sektor keuangan telah meningkatkan risiko financial fraud yang semakin kompleks, sementara pendekatan deteksi berbasis machine learning terpusat menghadapi keterbatasan privasi dan kepatuhan regulasi. Federated Learning (FL) menawarkan alternatif kolaborasi antar institusi tanpa berbagi data mentah, namun eksplorasi model berbasis tree dalam FL untuk deteksi fraud masih terbatas dan belum terdapat perbandingan sistematis dengan model berbasis gradient di bawah kondisi Non-IID dan class imbalance ekstrem. Penelitian ini mengevaluasi performa serta menganalisis explainability dari enam model FL dengan empat paradigma agregasi yang berbeda, yaitu Logistic Regression dan Support Vector Machine dengan FedAvg, Gradient Boosting Machine dengan best-model selection, FFD (1D-CNN) dan BERT (tabular Transformer) dengan accuracy-weighted FedAvg, serta FedXGBllr dengan tree ensemble aggregation. Eksperimen dilakukan menggunakan dataset PaySim pada simulasi cross-silo berbasis kerangka kerja Flower, dengan partisi Dirichlet untuk memodelkan heterogenitas data antar client dan penerapan SMOTE lokal untuk menangani class imbalance. Evaluasi menggunakan AUPRC sebagai metrik utama, didukung F1-score, Precision, dan Recall, sementara analisis explainability dilakukan melalui SHAP values per-client untuk mengukur stabilitas feature importance di bawah kondisi Non-IID. Penelitian ini diharapkan menghasilkan benchmark sistematis pertama yang mengintegrasikan empat paradigma agregasi FL pada konteks deteksi fraud finansial, bukti empiris pertama mengenai performa FedXGBllr pada kondisi class imbalance ekstrem dan Non-IID, serta kontribusi awal pada kajian Explainable Federated Learning melalui analisis stabilitas interpretasi SHAP antar paradigma model.

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
The digitalization of the financial sector has heightened the risk of increasingly complex financial fraud, while centralized machine learning approaches face limitations regarding privacy and regulatory compliance. Federated Learning (FL) offers an alternative for cross-institutional collaboration without sharing raw data; however, the exploration of tree-based models in FL for fraud detection remains limited, and no systematic comparison exists between gradient-based and tree-based models under Non-IID conditions and extreme class imbalance. This study evaluates the performance and analyzes the explainability of six FL models built upon four different aggregation paradigms, namely Logistic Regression and Support Vector Machine with FedAvg, Gradient Boosting Machine with best-model selection, FFD (1D-CNN) and BERT (tabular Transformer) with accuracy-weighted FedAvg, and FedXGBllr with tree ensemble aggregation. Experiments are conducted using the PaySim dataset in a cross-silo simulation on the Flower framework, with Dirichlet partitioning to model data heterogeneity across clients and local application of SMOTE to address class imbalance. Evaluation employs AUPRC as the primary metric, supported by F1-score, Precision, and Recall, while explainability analysis is conducted through per-client SHAP values to measure feature importance stability under Non-IID conditions. This study is expected to deliver the first systematic benchmark integrating four FL aggregation paradigms in financial fraud detection, the first empirical evidence of FedXGBllr performance under extreme class imbalance and Non-IID conditions, and an initial contribution to Explainable Federated Learning through stability analysis of SHAP interpretations across model paradigms.

#v(1em)
#par(first-line-indent: 0pt)[
  #text(weight: "bold")[Keywords:] Federated Learning, Financial Fraud Detection, FedXGBllr, Explainable AI, SHAP, Non-IID.
]

#pagebreak()

//=============================================================================
// 4. KATA PENGANTAR
//=============================================================================

#heading("KATA PENGANTAR", numbering: none)

// Typst 0.13+ removed par(indent: ...)
// Use explicit first-line indentation if needed: #indent[content]

#v(1.5em)

*Assalamu'alaikum warahmatullahi wabarakatuh.*

Puji dan syukur ke hadirat Allah SWT atas segala limpahan nikmat dan
rahmat-Nya sehingga penulis dapat menyelesaikan Tugas Akhir ini dengan judul "#title.id" sebagai salah satu syarat
untuk memperoleh gelar #program.title di #program.department, #program.university.

Penulisan Tugas Akhir ini tidak akan terlaksana dengan baik tanpa bimbingan, dukungan, dan motivasi dari berbagai pihak. Oleh karena itu, penulis ingin menyampaikan ucapan terima kasih yang sebesar-besarnya kepada:

+ *Bapak/Ibu Dosen Pembimbing 1* dan *Bapak/Ibu Dosen Pembimbing 2* atas bimbingan, saran, serta kesabaran dalam mengarahkan penulis selama proses penyusunan Tugas Akhir ini.
+ *Bapak/Ibu Penguji Sidang* yang telah memberikan masukan, kritik, dan saran konstruktif demi perbaikan karya ini.
+ Seluruh *partisipan penguji aplikasi* yang telah meluangkan waktu dan memberikan feedback berharga untuk pengembangan sistem ini.
+ Keluarga, teman-teman, dan rekan-rekan seperjuangan yang selalu memberikan dukungan moral dan semangat kepada penulis.

Penulis sangat menyadari bahwa penulisan Tugas Akhir ini masih jauh dari sempurna, oleh karena itu penulis mengharapkan kritik dan saran yang membangun untuk perbaikan di masa mendatang. Semoga hasil penelitian ini dapat bermanfaat bagi perkembangan ilmu pengetahuan dan teknologi.

Akhir kata, penulis mengharapkan semoga hasil dari penulisan dan penelitian
ini dapat memberikan informasi yang bermanfaat bagi para pembaca.

*Wassalamu'alaikum warahmatullahi wabarakatuh.*

#v(2em)
#align(right)[
  Surabaya, #dates.writingPeriod \
  \
  \
  \
  Penulis
]

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

// TODO: lengkapi lampiran dengan konfigurasi eksperimen final (hyperparameter, seed, environment) saat penyusunan buku Tugas Akhir.
Lampiran ini akan memuat rincian konfigurasi eksperimen, termasuk daftar hyperparameter lengkap, random seed, serta spesifikasi lingkungan komputasi yang digunakan dalam penelitian.

#pagebreak()

//=============================================================================
// BIOGRAFI PENULIS
//=============================================================================

#heading("BIOGRAFI PENULIS", numbering: none)

// TODO: tulis biografi penulis (tidak ada di Proposal TA v2.1; diisi saat penyusunan buku Tugas Akhir).
#grid(
  columns: (1fr, 2fr),
  column-gutter: 1em,
  block()[
    #image("resources/profile-picture.jpg")
  ],
  [
    *Amadeo Yesa*, lahir dan menempuh pendidikan sarjana di Program Studi S1 Teknik
    Informatika, Departemen Teknik Informatika, Fakultas Teknologi Elektro dan Informatika
    Cerdas, Institut Teknologi Sepuluh Nopember (ITS), Surabaya. Penulis memiliki minat pada
    bidang machine learning, federated learning, dan explainable artificial intelligence.
  ]
)
#block(
  inset: (left: 0em, top: -0.5em),
  [Penulis dapat dihubungi melalui surel untuk keperluan diskusi terkait Tugas Akhir ini.]
)
