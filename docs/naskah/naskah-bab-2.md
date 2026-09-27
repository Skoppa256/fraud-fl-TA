**Catatan transkripsi — bukan bagian naskah tesis.** Berkas ini adalah salinan kata demi kata dari naskah yang ditulis penulis, tanpa perbaikan apa pun. Kesalahan penomoran yang sudah diketahui (§1.4 dua kali, §3.2.1 dua kali, Tabel 3.6 dua kali, rujukan Tabel 3.9, dan dua ketik `Gamber`) sengaja dipertahankan, demikian pula kesalahan ketik lain seperti `darititik-titik`, `masing- masing`, `pohon- pohon`, dan `label aktual,dengan`. Penanda `<Tabel 3.1>` dan sejenisnya adalah penanda posisi objek dari penulis.

Berkas ini adalah **bagian 2 dari 3**. Bagian 1 = BAB 1, bagian 2 = BAB 2, bagian 3 = BAB 3.

---

# BAB 2
# TINJAUAN PUSTAKA

## 2.1 Hasil Penelitian Terdahulu

### 2.1.1 Deteksi Financial Fraud Berbasis Machine Learning dan Deep Learning Terpusat

Penelitian deteksi fraud secara terpusat telah berkembang pesat selama satu dekade terakhir yang terpusat pada arsitektur deep learning. Sharma dkk (2022) mengembangkan pendekatan credit card fraud detection berbasis Auto-Encoder yang digabungkan dengan klasifikasi deep neural network. Auto-encoder digunakan untuk mempelajari representasi laten dari pola transaksi normal sedangkan transaksi yang menunjukkan rekonstruksi dengan error tinggi diklasifikasikan sebagai anomali. Pendekatan ini berhasil meningkatkan kemampuan deteksi pada data dengan ketidakseimbangan kelas namun tetap dalam kondisi terpusat seluruh data transaksi dapat dikumpulkan pada satu titik komputasi.

Pengembangan lebih lanjut dilakukan oleh Baghdadi dkk (2024) yang mengusulkan pendekatan ensemble learning yang menggabungkan energy-based Restricted Boltzmann Machine (RBM) dengan extended Long Short-Term Memory (xLSTM) untuk predictive analytics pada deteksi fraud kartu kredit. Hasil eksperimen menunjukkan bahwa kombinasi pembelajaran representasi berbasis energi dengan arsitektur sekuensial mampu menangkap pola temporal yang kompleks pada transaksi finansial. Meskipun kedua penelitian tersebut menunjukkan keunggulan performa, keduanya beroperasi dalam kondisi terpusat sehingga tidak menjawab persoalan privasi dan kepatuhan regulasi yang menjadi penting dalam konteks lintas institusi. Selain itu, model-model berbasis deep learning tersebut bersifat black-box sehingga aspek interpretabilitas menjadi sulit dipenuhi tanpa metode XAI tambahan.

### 2.1.2 Penerapan Federated Learning untuk Deteksi Financial Fraud

Sebagai jawaban atas keterbatasan kondisi terpusat atau centralized, sejumlah peneliti mulai mengeksplorasi penerapan Federated Learning untuk deteksi financial fraud. Suvarna dan Meena Kowshalya (2020) merupakan salah satu kontributor awal yang mendemonstrasikan penerapan FL pada credit card fraud detection. Penelitian tersebut menggunakan model konvensional yang dilatih secara federated dengan FedAvg dan menunjukkan bahwa pendekatan FL mampu memberikan performa yang sebanding dengan pendekatan terpusat dengan tetap menjaga privasi data pelanggan. Namun, penelitian tersebut belum mempertimbangkan kompleksitas distribusi data Non-IID yang merupakan kondisi nyata pada kolaborasi antar institusi keuangan.

Penelitian yang lebih lanjut oleh Venkata Krishna Reddy dkk (2024) memperluas eksplorasi tersebut dengan mengintegrasikan arsitektur deep learning ke dalam kerangka FL untuk deteksi fraud kartu kredit. Pendekatan ini berhasil meningkatkan akurasi deteksi melalui pemanfaatan kapasitas representasi deep neural network namun masih memiliki kelemahan model deep learning dalam hal interpretabilitas dan tetap mengandalkan agregasi berbasis FedAvg yang sensitif terhadap heterogenitas data.

Perkembangan paling menjanjikan datang dari Tang dan Liang (2024) yang mengusulkan kerangka federated graph learning untuk deteksi fraud kartu kredit. Penelitian tersebut memanfaatkan struktur graf untuk memodelkan hubungan antar entitas transaksi, kemudian melatih Graph Neural Network (GNN) secara federated di atas kerangka tersebut. Hasilnya menunjukkan bahwa pendekatan berbasis graf mampu menangkap pola fraud yang bersifat relasional yang sulit dideteksi oleh model konvensional. Walaupun demikian, pendekatan ini menambahkan kompleksitas komputasi yang signifikan dan tetap menggunakan paradigma agregasi berbasis gradient sehingga belum mengeksplorasi kemungkinan penggunaan model berbasis tree yang justru terbukti unggul untuk data tabular transaksional.

Aljunaid dkk (2025) memberikan kontribusi penting dengan mengusulkan kerangka FL berbasis XAI untuk deteksi fraud perbankan. Penelitian tersebut menggunakan tiga model konvensional, yaitu Logistic Regression (LR), Support Vector Machine (SVM), dan Gradient Boosting Machine (GBM) dengan skema agregasi best-model selection, yaitu pemilihan model dengan akurasi terbaik di antara seluruh client. Hasilnya menunjukkan bahwa model GBM mencapai performa terbaik dan integrasi SHAP berhasil memberikan transparansi terhadap keputusan model. Meskipun pendekatan ini telah memperkenalkan dimensi explainability ke dalam FL, penelitian tersebut belum membandingkan hasilnya dengan model FL berbasis tree yang lebih modern seperti FedXGBllr, dan belum mengkaji secara spesifik bagaimana skema agregasi yang berbeda memengaruhi stabilitas interpretasi SHAP di bawah kondisi Non-IID.

### 2.1.3 Integrasi Model Berbasis Tree ke dalam Kerangka Federated Learning

Eksplorasi model berbasis tree dalam ekosistem FL relatif terbatas dibandingkan model berbasis gradient. Hal ini disebabkan oleh karakteristik struktur pohon berupa aturan percabangan yang dapat berbeda antar client sehingga skema agregasi standar seperti FedAvg tidak dapat diterapkan secara langsung pada struktur tersebut. Ma dkk (2023) mengusulkan FedXGBllr (Federated XGBoost with Learnable Learning Rates), sebuah kerangka pelatihan XGBoost secara federated dalam horizontal setting yang tidak bergantung pada pertukaran gradient dan hessian antar client. Setiap client melatih tree ensemble secara lokal, kemudian server mengagregasi seluruh tree ensemble dan melatih one-layer 1D Convolutional Neural Network (CNN) untuk mempelajari learning rate setiap pohon secara global. Pendekatan ini terbukti menurunkan communication overhead hingga 25–700 kali lipat dibandingkan metode sebelumnya seperti SimFL sekaligus menghilangkan risiko kebocoran privasi melalui gradient.

Pada tahap berikutnya, output dari kumpulan pohon tersebut menjadi input bagi one-layer 1D Convolutional Neural Network (CNN) untuk mempelajari kontribusi pohon terhadap prediksi akhir. Dengan demikian, komponen CNN pada FedXGBllr bekerja atas representasi output pohon dan bukan atas urutan transaksi nasabah. Pendekatan ini memungkinkan penggabungan informasi dari tree ensemble lokal tanpa merata-ratakan struktur pohon secara langsung (Ma dkk, 2023).

Walaupun demikian, evaluasi original FedXGBllr dilakukan pada dataset publik berskala umum seperti HIGGS, SUSY, dan a9a yang tidak merefleksikan karakteristik domain finansial. Dataset-dataset tersebut tidak memiliki class imbalance ekstrem maupun struktur fitur transaksional yang khas pada deteksi fraud. Selain itu, evaluasi original menggunakan partisi data yang seimbang dan tidak menguji performa pada skenario Non-IID berbasis distribusi Dirichlet yang lebih realistis. Aspek explainability model juga belum dieksplorasi dalam penelitian Ma dkk (2023) dimana FedXGBllr memiliki potensi interpretasi melalui struktur pohon yang transparan dan bobot learning rate yang dapat dipelajari.

### 2.1.4 Explainable AI untuk Deteksi Financial Fraud

Aspek explainability menjadi krusial dalam domain keuangan karena keputusan model harus dapat dipertanggungjawabkan kepada auditor, regulator, dan nasabah yang terdampak. Doshi-Velez dan Kim (2017) memberikan fondasi konseptual mengenai pentingnya interpretable machine learning dengan mengusulkan kerangka evaluasi yang sistematis untuk mengukur kualitas interpretasi. Mereka menekankan bahwa interpretabilitas tidak hanya bermanfaat untuk transparansi, tetapi juga untuk identifikasi bias, validasi domain pengetahuan, serta peningkatan kepercayaan pengguna terhadap sistem berbasis ML.

Bussmann dkk (2021) mengaplikasikan kerangka tersebut pada manajemen risiko kredit dengan memanfaatkan SHAP (SHapley Additive exPlanations) untuk menjelaskan keputusan model XGBoost. Hasil penelitian mereka menunjukkan bahwa SHAP mampu memberikan feature importance yang konsisten dan dapat diaudit sehingga cocok untuk diadopsi pada domain finansial yang ketat regulasinya. Penelitian ini menjadi pijakan penting bahwa kombinasi model berbasis tree dengan SHAP merupakan pasangan yang efektif untuk skenario keuangan.

Aljunaid dkk (2025) telah memperkenalkan integrasi XAI ke dalam kerangka FL untuk deteksi fraud perbankan. Namun, analisis SHAP yang mereka lakukan masih bersifat agregat dan belum menganalisis bagaimana variasi distribusi data antar client memengaruhi konsistensi feature importance. Oleh karena itu, penelitian ini menganalisis stabilitas interpretasi SHAP pada kondisi IID dan Non-IID dengan membandingkan model parametrik, model berbasis tree, dan model deep learning. Perbandingan tersebut mencakup FedAvg untuk LR dan SVM, best-model selection untuk GBM, accuracy-weighted FedAvg untuk FFD dan BERT, serta tree ensemble aggregation dengan learnable learning rates untuk FedXGBllr.

### 2.1.5 Rangkuman Penelitian Terdahulu

Berdasarkan kajian di atas, penelitian ini menganalisis penerapan FL untuk deteksi financial fraud dengan class imbalance ekstrem dan distribusi Non-IID, integrasi model berbasis tree, khususnya FedXGBllr yang bersifat gradient-less, ke dalam ekosistem FL untuk sektor keuangan, dan stabilitas interpretasi SHAP antar paradigma agregasi FL. Tabel 2.1 merangkum hasil penelitian terdahulu yang dikaji beserta identifikasi celah riset yang menjadi dasar penelitian ini.

`<Tabel 2.1 Rangkuman Hasil Penelitian Terdahulu>`

## 2.2 Dasar Teori

### 2.2.1 Financial Fraud Detection

Financial fraud didefinisikan sebagai tindakan disengaja yang dilakukan oleh seseorang atau kelompok untuk memperoleh keuntungan finansial secara tidak sah melalui penipuan, manipulasi, atau penyalahgunaan kepercayaan dalam sistem keuangan (Hilal dkk, 2022). Bentuk fraud yang umum meliputi penyalahgunaan kartu kredit, money laundering, identity theft, dan transaksi yang mencurigakan. Karakteristik utama yang membedakan deteksi fraud dari permasalahan klasifikasi lainnya adalah ketidakseimbangan kelas yang ekstrem (extreme class imbalance) di mana proporsi transaksi fraud terhadap transaksi normal seringkali kurang dari 1%.

Pendekatan deteksi fraud secara umum dapat dikelompokkan menjadi tiga kategori. Pendekatan rule-based menggunakan aturan yang ditetapkan oleh ahli domain namun kurang adaptif terhadap pola fraud baru. Pendekatan machine learning memanfaatkan algoritma pembelajaran untuk mengidentifikasi pola dari data historis transaksi. Adapun pendekatan deep learning menggunakan arsitektur jaringan saraf dalam untuk menangkap pola non-linear yang kompleks. Penelitian ini membandingkan model parametrik, model berbasis tree, dan model deep learning untuk mengamati perbedaan kemampuan deteksi serta karakteristik interpretasinya pada data tabular finansial.

### 2.2.2 Federated Learning

Federated Learning (FL) adalah paradigma pembelajaran mesin terdistribusi yang memungkinkan beberapa pihak (client) melatih model bersama tanpa memindahkan data mentah dari masing-masing pemiliknya (McMahan dkk, 2017). Pelatihan dilakukan secara lokal di setiap client, dan hanya parameter atau pembaruan model yang dikomunikasikan ke server pusat untuk diagregasi menjadi model global. Mekanisme tersebut membatasi perpindahan data mentah selama pelatihan, meskipun pertukaran parameter atau model tidak dengan sendirinya memberikan jaminan privasi terhadap seluruh bentuk serangan.

`<Gamber 2.1 Arsitektur Umum Federated Learning. Sumber: Sabuhi dkk (2024)>`

Berdasarkan distribusi data, FL diklasifikasikan menjadi tiga jenis utama, yaitu Horizontal Federated Learning (HFL) ketika client memiliki ruang fitur yang sama tetapi sampel yang berbeda, Vertical Federated Learning (VFL) ketika client memiliki ruang sampel yang sama tetapi fitur yang berbeda, dan Federated Transfer Learning (FTL) ketika ruang fitur dan ruang sampel berbeda. Penelitian ini menggunakan paradigma HFL karena setiap institusi keuangan diasumsikan memiliki struktur fitur transaksi yang sama tetapi sampel nasabah yang berbeda.

Berdasarkan skala dan jumlah client, FL juga dibedakan menjadi cross-device FL (jutaan perangkat seperti telepon genggam) dan cross-silo FL (puluhan organisasi seperti bank atau rumah sakit). Penelitian ini berada pada konteks cross-silo FL dengan asumsi honest-but-curious clients, yaitu client mengikuti protokol pelatihan dengan jujur namun mungkin mencoba menyimpulkan informasi dari pesan yang diterima. Proses pelatihan FL secara umum mengikuti empat tahapan iteratif (Kairouz dkk, 2021):

1. Inisialisasi: Server menginisialisasi model global 𝑤0 dan mendistribusikannya kepada seluruh client.
2. Pelatihan Lokal: Setiap client 𝑘 melatih model menggunakan data lokalnya 𝐷𝑘 untuk menghasilkan model lokal 𝑤𝑘𝑡.
3. Agregasi: Client mengirimkan parameter atau pembaruan model ke server, yang kemudian mengagregasi seluruh kontribusi menjadi model global 𝑤𝑡+1.
4. Iterasi: Model global terbaru dikirim kembali ke client untuk putaran berikutnya hingga konvergensi tercapai atau jumlah putaran maksimum terpenuhi.

Adapun Federated Averaging (FedAvg) yang diperkenalkan oleh McMahan dkk (2017) merupakan algoritma agregasi paling fundamental dalam FL. FedAvg melakukan rata-rata berbobot terhadap parameter model dari seluruh client berdasarkan ukuran data lokal masing- masing, sebagaimana dirumuskan pada Persamaan (2.1):

`<Persamaan 2.1>`

dengan 𝑤𝑡+1 sebagai parameter global pada putaran ke-(𝑡 + 1), 𝐾 sebagai jumlah client, 𝑛𝑘 sebagai jumlah sampel pada client ke-𝑘, 𝑛 = ∑ 𝑛𝑘 sebagai total sampel, dan 𝑤𝑘𝑡 sebagai parameter lokal client ke-𝑘 pada putaran ke-𝑡. FedAvg cocok untuk model parametrik yang dilatih dengan gradient descent seperti Logistic Regression dan Support Vector Machine namun tidak dapat diterapkan secara natural untuk model berbasis tree karena struktur pohon tidak dapat dirata-ratakan secara element-wise.

Untuk model yang tidak kompatibel dengan agregasi berbobot seperti FedAvg, Aljunaid dkk (2025) memperkenalkan skema best-model selection yang merumuskan agregasi sebagai pemilihan model dengan kinerja terbaik di antara seluruh client, sebagaimana dirumuskan pada Persamaan (2.2):

`<Persamaan 2.2>`

dengan 𝑊∗ sebagai bobot model global terpilih, 𝑊𝑖 sebagai bobot model dari client ke-𝑖, 𝐴(⋅) sebagai fungsi evaluasi kinerja (misalnya akurasi atau AUPRC), dan 𝑉𝑖 sebagai validation set. Skema ini cocok untuk model Gradient Boosting Machine (GBM) yang strukturnya tidak dapat dirata-ratakan dengan trade-off berupa penyederhanaan agregasi dan potensi kehilangan informasi dari client yang tidak terpilih.

Selain ukuran data lokal, kontribusi client dalam agregasi dapat juga mempertimbangkan kinerja model lokal. Pada varian accuracy-weighted FedAvg yang digunakan dalam penelitian ini, AUPRC lokal digunakan sebagai komponen pembobotan bersama ukuran data client. Dengan demikian, istilah accuracy-weighted pada nama skema tidak merujuk pada penggunaan metrik accuracy sebagai bobot. Penyesuaian ini relevan pada data dengan class imbalance karena accuracy yang tinggi masih dapat diperoleh oleh model yang gagal mendeteksi kelas fraud.

Perbandingan antar paradigma agregasi juga perlu mempertimbangkan model yang digunakan. FedAvg, best-model selection, dan agregasi tree ensemble diterapkan pada struktur model yang berbeda. Maka dari itu, perbedaan performa antar kelompok dalam penelitian ini menggambarkan konfigurasi model dan agregasinya secara bersamaan sehingga tidak seluruhnya dapat diatribusikan pada paradigma agregasi saja.

### 2.2.3 Model Parametrik (Gradient-Based)

Logistic Regression adalah model klasifikasi linear yang memodelkan probabilitas keluaran biner menggunakan fungsi sigmoid. Model ini bekerja dengan mempelajari sebuah vektor bobot 𝑤 yang merepresentasikan pengaruh setiap fitur terhadap kemungkinan kelas tertentu. Hasil kombinasi linear antara fitur masukan dan bobot kemudian ditransformasikan oleh fungsi sigmoid agar menghasilkan probabilitas pada rentang 0 hingga 1.

Pelatihan LR dilakukan dengan meminimalkan binary cross-entropy loss dan parameternya berupa vektor koefisien serta bias sehingga kompatibel dengan agregasi FedAvg. Implementasi penelitian ini menggunakan solver lbfgs yang merupakan metode quasi-Newton bermemori terbatas dan bukan gradient descent. Keluaran pelatihannya tetap berupa koefisien linear yang dapat dirata-ratakan antar client. Keunggulan utama LR adalah interpretabilitas koefisien yang langsung menunjukkan arah dan besarnya pengaruh setiap fitur. Namun, LR memiliki keterbatasan dalam menangkap interaksi non-linear antar fitur sehingga kurang optimal untuk pola fraud yang kompleks.

Adapun Support Vector Machine (SVM) merupakan algoritma klasifikasi yang mencari hyperplane pemisah optimal dengan margin maksimum antara dua kelas (Cortes dan Vapnik, 1995). Secara intuitif, SVM berusaha menggambar garis pemisah yang sejauh mungkin darititik-titik data terdekat di kedua kelas, sehingga model lebih robust terhadap data baru.

Pada pelatihan SVM, terdapat parameter regularisasi yang mengontrol trade-off antara lebar margin dan toleransi terhadap kesalahan klasifikasi. Untuk data yang tidak dapat dipisahkan secara linear, SVM dapat diperluas dengan kernel trick yang memetakan data ke ruang berdimensi lebih tinggi. Pada konteks penelitian ini, SVM linear digunakan agar parameternya dapat diagregasi menggunakan FedAvg.

### 2.2.4 Model Berbasis Tree

Gradient Boosting Machine (GBM) adalah model ensemble yang membangun pohon keputusan secara aditif dan berurutan dengan setiap pohon dilatih untuk memperbaiki kesalahan pohon sebelumnya. Prediksi akhir model adalah jumlah berbobot dari prediksi seluruh pohon dalam ensemble dengan kontribusi setiap pohon dikontrol oleh learning rate. Mekanisme ini memungkinkan GBM membangun model yang kuat dari banyak weak learner berupa pohon- pohon dangkal.

GBM unggul dalam menangani fitur heterogen, interaksi non-linear, dan nilai missing sehingga sangat cocok untuk data tabular transaksional. Namun, struktur pohonnya yang berupa urutan kondisi percabangan tidak dapat dirata-ratakan secara langsung sehingga GBM memerlukan skema agregasi alternatif seperti best-model selection dalam konteks FL.

Extreme Gradient Boosting (XGBoost) merupakan implementasi GBM yang dioptimalkan oleh Chen dan Guestrin (2016) dengan penambahan regularisasi eksplisit, dukungan komputasi paralel, dan penanganan missing value yang efisien. XGBoost menambahkan komponen regularisasi pada fungsi loss-nya untuk mengontrol kompleksitas pohon sehingga model lebih tahan terhadap overfitting dibandingkan GBM klasik.

Untuk membangun setiap pohon, XGBoost menggunakan informasi turunan pertama dan turunan kedua dari fungsi loss untuk mengevaluasi kualitas setiap kandidat split di tiap node. Pendekatan berbasis turunan inilah yang membuat XGBoost sangat efisien namun sekaligus menjadi tantangan utama saat hendak diintegrasikan ke dalam FL karena dalam skema federated tradisional, client harus saling bertukar informasi turunan tersebut, yang berimplikasi pada frekuensi komunikasi yang sangat tinggi dan potensi kebocoran privasi karena informasi turunan dapat dieksploitasi untuk merekonstruksi data pelatihan (Zhu dkk, 2019).

FedXGBllr yang diusulkan oleh Ma dkk (2023) merupakan kerangka horizontal federated XGBoost yang dirancang untuk mengatasi keterbatasan XGBoost konvensional dalam FL. Inovasi utamanya terletak pada sifat gradient-less dimana client tidak perlu bertukar informasi turunan apapun melainkan hanya mengirimkan tree ensemble yang sudah jadi.

`<Gambar 2.2 …>`

Mekanismenya terdiri dari dua tahap utama. Tahap pertama, setiap client melatih XGBoost tree ensemble lokal menggunakan datanya sendiri kemudian mengirimkan seluruh tree ensemble tersebut ke server. Server lalu menggabungkan seluruh tree ensemble dari semua client menjadi satu kumpulan pohon agregat yang besar.

Tahap kedua, server tidak hanya menggabungkan prediksi pohon-pohon tersebut secara naif melainkan mempelajari bobot kontribusi (learning rate) untuk setiap pohon melalui sebuah one-layer 1D Convolutional Neural Network (CNN) yang dilatih secara federated dengan FedAvg. Dengan kata lain, FedXGBllr membiarkan setiap client berkontribusi melalui struktur pohonnya namun seberapa besar pengaruh setiap pohon terhadap prediksi akhir ditentukan secara adaptif oleh CNN tersebut.

Karakteristik gradient-less pada FedXGBllr menghilangkan kebutuhan pertukaran informasi turunan dalam pembentukan ensemble tetapi tidak dapat diartikan sebagai jaminan bahwa model yang dipertukarkan bebas dari seluruh risiko kebocoran informasi. Selain itu, jumlah putaran komunikasi tidak bergantung pada kedalaman atau jumlah pohon sehingga communication overhead berkurang secara signifikan hingga 25–700 kali lebih efisien dibandingkan metode FL berbasis tree sebelumnya seperti SimFL.

### 2.2.5 Model Deep Learning

Model deep learning mempelajari representasi fitur melalui beberapa lapisan transformasi non-linear. Pada data tabular, struktur masukan tersebut berbeda dari citra maupun teks karena setiap kolom dapat memiliki arti dan skala yang berbeda. Oleh karena itu, arsitektur model perlu disesuaikan dengan bentuk fitur yang digunakan dalam penelitian.

Model FFD pada penelitian ini menggunakan 1D Convolutional Neural Network. Operasi konvolusi diterapkan pada representasi fitur transaksi untuk membentuk pola yang kemudian digunakan dalam klasifikasi fraud. Penggunaan konvolusi satu dimensi tersebut tidak menunjukkan bahwa penelitian memodelkan urutan waktu transaksi. Prediksi tetap diberikan pada setiap baris data tabular yang menjadi masukan model.

Adapun model yang diberi nama BERT pada eksperimen merupakan tabular Transformer berbasis FT-Transformer dan bukan model bahasa BERT yang dilatih pada dataset teks. FT-Transformer mengembangkan representasi token dari fitur tabular dan menggunakan self-attention untuk mempelajari hubungan antar fitur (Gorishniy dkk, 2021). Nama BERT dipertahankan agar konsisten dengan identitas model dalam hasil eksperimen sedangkan istilah FT-Transformer digunakan untuk menjelaskan arsitekturnya.

### 2.2.6 Distribusi Data Non-IID dan Partisi Dirichlet

Pada lingkungan FL, data antar client umumnya bersifat non-independent and identically distributed (Non-IID) yang berarti setiap client memiliki distribusi data yang berbeda. Kondisi ini mencerminkan situasi nyata pada institusi keuangan dimana setiap bank memiliki segmentasi nasabah, profil risiko, dan pola transaksi yang khas. Heterogenitas data antar client terbukti menurunkan performa konvergensi model FL terutama pada algoritma agregasi seperti FedAvg yang mengasumsikan distribusi data relatif seragam (Li dkk, 2021).

Untuk mensimulasikan kondisi Non-IID secara terkontrol dan dapat direplikasi, partisi berbasis distribusi Dirichlet umum digunakan dalam literatur FL (Hsu dkk, 2019). Secara intuitif, distribusi Dirichlet dengan parameter konsentrasi 𝛼 mengontrol seberapa heterogen distribusi label antar client. Nilai 𝛼 kecil menghasilkan distribusi yang sangat heterogen di mana setiap client cenderung hanya memiliki sampel dari beberapa kelas saja sehingga merepresentasikan kondisi Non-IID yang ekstrem. Sebaliknya, nilai 𝛼 besar menghasilkan distribusi yang mendekati IID di mana proporsi setiap kelas relatif serupa antar client.

`<Gambar 2.3 …>`

### 2.2.7 Synthetic Minority Oversampling Technique (SMOTE)

SMOTE merupakan teknik penanganan class imbalance yang menghasilkan sampel sintetis dari kelas minoritas untuk menyeimbangkan distribusi kelas (Chawla dkk, 2002). Berbeda dengan teknik oversampling sederhana yang hanya menduplikasi sampel minoritas, SMOTE membuat sampel baru dengan cara melakukan interpolasi antara satu sampel minoritas dan tetangga terdekatnya di ruang fitur. Hasilnya adalah sampel sintetis yang berada di antara dua sampel asli, sehingga lebih variatif dibandingkan sekadar duplikasi.

Pada penelitian ini, SMOTE diterapkan secara lokal pada setiap client sebelum proses pelatihan federated dimulai, agar prinsip privasi FL tetap terjaga. Penerapan SMOTE secara global akan mengharuskan agregasi data mentah ke satu titik komputasi, yang melanggar paradigma FL.

`<Gamber 2.4 …>`

Efektivitas SMOTE bergantung pada sebaran sampel kelas minoritas di ruang fitur. Pembentukan sampel melalui interpolasi didasarkan pada anggapan bahwa area di antara sampel minoritas yang berdekatan dapat merepresentasikan kelas yang sama. Namun, SMOTE tidak secara khusus membedakan sampel yang mewakili pola umum kelas minoritas dari outlier. Apabila interpolasi melibatkan outlier atau melewati wilayah kelas mayoritas, sampel sintetis yang dihasilkan berpotensi memperbesar tumpang tindih antarkelas dan menyulitkan pembentukan batas keputusan. Oleh karena itu, penambahan sampel minoritas melalui SMOTE tidak selalu menghasilkan peningkatan performa klasifikasi karena manfaatnya turut ditentukan oleh struktur kelas pada data.

Selain sebaran kelas, ketersediaan sampel minoritas asli juga perlu diperhatikan. Weiss (2004) membedakan kelangkaan relatif (relative rarity), yaitu rendahnya proporsi suatu kelas dibandingkan kelas lainnya, dan kelangkaan absolut (absolute rarity), yaitu terbatasnya jumlah sampel yang tersedia untuk mempelajari karakteristik kelas tersebut. Dalam konteks SMOTE, perbedaan ini menunjukkan bahwa penambahan jumlah sampel sintetis tidak dapat disamakan dengan penambahan pengamatan nyata yang independen. Karena sampel sintetis dibentuk berdasarkan sampel yang telah tersedia, keragaman pola yang dapat direpresentasikan tetap bergantung pada cakupan sampel minoritas asli. Dengan demikian, distribusi kelas yang lebih seimbang setelah oversampling belum menjamin bahwa karakteristik kelas minoritas telah terwakili secara memadai.

### 2.2.8 Metrik Evaluasi untuk Imbalanced Classification

Pada konteks imbalanced classification, metrik accuracy tidak dapat diandalkan karena bias terhadap kelas mayoritas. Penelitian ini menggunakan AUPRC sebagai metrik utama, disertai Recall@5%FPR untuk mengukur kemampuan deteksi pada batas false positive rate tertentu. Precision, Recall, dan F1-score digunakan sebagai metrik pelengkap (Saito dan Rehmsmeier, 2015).

Confusion Matrix. Matriks ini menyajikan empat komponen dasar evaluasi: True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN), di mana kelas positif merepresentasikan transaksi fraud.

Precision mengukur proporsi prediksi positif yang benar (Persamaan (2.3)):

`<Persamaan 2.3>`

Recall mengukur proporsi sampel positif yang berhasil dideteksi (Persamaan (2.4)):

`<Persamaan 2.4>`

F1-score merupakan rata-rata harmonik dari Precision dan Recall (Persamaan (2.5)):

`<Persamaan 2.5>`

AUPRC (Area Under the Precision-Recall Curve) merangkum kemampuan model dalam mendeteksi kelas positif pada berbagai ambang klasifikasi. Perubahan ambang menghasilkan pasangan nilai Precision dan Recall yang berbeda yang kemudian membentuk kurva Precision-Recall. Kurva ini menggambarkan hubungan antara banyaknya kasus positif yang berhasil ditemukan dan ketepatan prediksi positif model. Dalam penelitian ini, AUPRC dihitung menggunakan average precision, yaitu penjumlahan nilai Precision yang masing-masing dibobot berdasarkan besarnya peningkatan Recall. Dengan demikian, model memperoleh nilai yang tinggi apabila mampu menemukan lebih banyak kasus positif sambil mempertahankan Precision yang tinggi. Perhitungan ini berbeda dari metode trapezoidal yang menghitung luas dengan menghubungkan titik-titik kurva menggunakan garis lurus. Saito dan Rehmsmeier (2015) menunjukkan bahwa AUPRC lebih informatif dibandingkan AUC-ROC pada data dengan class imbalance ekstrem, karena AUC-ROC cenderung optimistis ketika kelas negatif jauh lebih banyak dari kelas positif. Oleh karena itu, AUPRC dipilih sebagai metrik utama dalam penelitian ini.

Recall@5%FPR mengukur proporsi kasus fraud yang berhasil dideteksi ketika false positive rate (FPR) berada pada tingkat 5%. FPR merupakan proporsi transaksi sah yang keliru diklasifikasikan sebagai fraud, sebagaimana ditunjukkan pada Persamaan (2.6). Sebagai contoh, nilai Recall@5%FPR sebesar 0,80 berarti model mampu mendeteksi 80% kasus fraud dengan tingkat kesalahan penandaan sebesar 5% dari seluruh transaksi sah.

`<Persamaan 2.6>`

Jika AUPRC merangkum performa model pada berbagai ambang klasifikasi, Recall@5%FPR menunjukkan kemampuan deteksi pada tingkat kesalahan yang sama. Metrik ini membantu menilai seberapa banyak fraud yang dapat ditemukan ketika kesalahan deteksi terhadap transaksi sah dibatasi. Oleh karena itu, Recall@5%FPR digunakan sebagai metrik pelengkap untuk membandingkan kemampuan deteksi model pada kondisi operasional yang lebih spesifik.

Kalibrasi menunjukkan kesesuaian antara probabilitas yang diprediksi model dan frekuensi kejadian yang sebenarnya. Sebagai contoh, pada kumpulan transaksi dengan prediksi probabilitas fraud sekitar 20%, model yang terkalibrasi akan menunjukkan proporsi fraud aktual sekitar 20%. Keluaran berupa probabilitas belum tentu terkalibrasi sehingga aspek ini perlu dievaluasi tersendiri. Brier score mengukur rata-rata kuadrat selisih antara probabilitas prediksi dan label aktual,dengan nilai yang lebih rendah menunjukkan kesalahan prediksi probabilitas yang lebih kecil. Namun, metrik ini turut dipengaruhi kemampuan diskriminasi model sehingga bukan ukuran kalibrasi semata. Evaluasi dapat dilengkapi dengan calibration slope dan calibration-in-the-large (Van Calster dkk, 2016). Slope memiliki nilai ideal 1, nilai di bawah 1 menunjukkan prediksi yang terlalu ekstrem, sedangkan nilai di atas 1 menunjukkan prediksi yang kurang bervariasi. Calibration-in-the-large memiliki nilai ideal 0, dengan nilai negatif menunjukkan kecenderungan probabilitas terlalu tinggi dan nilai positif menunjukkan kecenderungan probabilitas terlalu rendah.

Ambang klasifikasi merupakan batas skor yang digunakan untuk menentukan apakah suatu transaksi dikategorikan sebagai fraud. Perubahan ambang dapat mengubah Precision, Recall, dan F1-score meskipun skor prediksi model tetap sama. Penurunan ambang memungkinkan lebih banyak fraud terdeteksi,tetapi juga dapat meningkatkan jumlah transaksi sah yang keliru ditandai. Sebaliknya, AUPRC merangkum kurva Precision-Recall pada berbagai ambang sehingga tidak bergantung pada pemilihan satu ambang tertentu. Oleh karena itu, evaluasi perlu membedakan kemampuan model dalam mengurutkan transaksi berdasarkan risiko, ketepatan probabilitas yang dihasilkan, dan hasil klasifikasi pada ambang yang dipilih.

### 2.2.9 Explainable Artificial Intelligence (XAI) dan SHAP

Explainable Artificial Intelligence (XAI) merujuk pada serangkaian metode dan teknik yang bertujuan membuat keputusan model machine learning dapat dipahami oleh manusia (Doshi-Velez dan Kim, 2017). Pada domain keuangan, explainability tidak hanya berfungsi sebagai sarana validasi teknis tetapi juga sebagai prasyarat regulasi dan transparansi terhadap auditor, regulator, dan nasabah. Doshi-Velez dan Kim (2017) mengusulkan tiga taksonomi evaluasi explainability, yaitu application-grounded yang melibatkan evaluasi pada aplikasi nyata oleh praktisi domain, human-grounded yang menggunakan tugas eksperimental dengan partisipan manusia, dan functionally-grounded yang menggunakan proksi formal tanpa keterlibatan manusia.

SHAP (Lundberg dan Lee, 2017) adalah kerangka untuk interpretasi prediksi model yang berlandaskan teori permainan kooperatif. Bagi setiap fitur 𝑗 pada sampel 𝑥, Shapley value 𝜑𝑗 mengukur kontribusi rata-rata fitur tersebut terhadap prediksi model dibandingkan dengan rata-rata prediksi baseline. Nilai Shapley dirumuskan pada Persamaan (2.7):

`<Persamaan 2.7>`

dengan 𝐹 sebagai himpunan seluruh fitur, 𝑆 sebagai subset fitur tanpa fitur 𝑗, dan 𝑓𝑥(𝑆) sebagai nilai prediksi ketika fitur dalam 𝑆 dipertahankan serta fitur lainnya diperlakukan menurut distribusi referensi. Distribusi referensi tersebut disebut background distribution. Pemilihannya memengaruhi nilai dasar dan kontribusi fitur sehingga penjelasan SHAP selalu perlu dibaca dalam hubungannya dengan referensi yang digunakan.

Komputasi SHAP memerlukan dua himpunan data yang berbeda peran. Yang pertama adalah background distribution, yaitu sampel referensi yang digunakan untuk mengaproksimasi nilai model 𝐸[𝑓(𝑧)] ketika subset fitur tertentu diasumsikan tidak teramati. Background distribution secara konseptual merepresentasikan distribusi data "normal" yang menjadi acuan baseline interpretasi. Yang kedua adalah explanation data, yaitu himpunan sampel yang akan dijelaskan kontribusi fiturnya melalui Shapley values 𝜑𝑗. Pemilihan kedua himpunan ini memengaruhi validitas interpretasi dimana background yang tidak representatif menghasilkan baseline yang bias sedangkan explanation data yang berbeda antar konteks evaluasi membuat hasil interpretasi sulit dibandingkan secara langsung.

Properti local accuracy pada SHAP menyatakan bahwa jumlah kontribusi fitur dan nilai dasar menghasilkan kembali keluaran model yang dijelaskan. Namun, pemenuhan hubungan penjumlahan tersebut saja belum membuktikan bahwa seluruh kontribusi fitur telah dihitung secara eksak. Dua himpunan atribusi yang berbeda masih dapat menghasilkan jumlah yang sama. Pemeriksaan penjelasan karena itu perlu mempertimbangkan jenis explainer dan variasi estimasinya. Contoh penyajian atribusi SHAP dalam bentuk summary plot ditunjukkan pada Gambar 2.5.

`<Gambar 2.5 …>`

LinearSHAP digunakan untuk model linear sedangkan TreeSHAP memanfaatkan struktur model berbasis pohon. Algoritma TreeSHAP diperkenalkan oleh Lundberg dkk (2019) dan menghitung Shapley value pada model pohon dalam waktu polinomial. Keduanya memberikan pengukuran deterministik untuk model, masukan, dan referensi yang tetap. Adapun KernelSHAP digunakan karena penjelasan diberikan terhadap keseluruhan fungsi prediksi model. KernelSHAP menggunakan regresi berbobot atas koalisi fitur dan dapat melibatkan pengambilan sampel koalisi ketika seluruh kombinasi tidak dienumerasi.

Pada penjelasan linear interventional, kontribusi fitur mengikuti hubungan 𝜑𝑗(𝑥) = 𝑤𝑗(𝑥𝑗− 𝜇𝑗) dengan 𝑤𝑗 sebagai koefisien model dan 𝜇𝑗 sebagai rata-rata fitur pada background. Dengan demikian, perubahan rata-rata background dapat mengubah atribusi meskipun koefisien model tetap. Adapun pada model non-linear, perubahan bentuk distribusi background juga dapat memengaruhi penjelasan. Perbedaan tersebut perlu dipertimbangkan ketika membandingkan stabilitas antar keluarga model.

Untuk merangkum penjelasan sejumlah sampel, feature importance dihitung melalui rata-rata nilai absolut SHAP. Bagi client 𝑐 dan fitur 𝑗, nilai tersebut dinyatakan pada Persamaan (2.8):

`<Persamaan 2.8>`

dengan 𝑁𝑐 sebagai jumlah sampel yang dijelaskan pada client 𝑐. Penggunaan nilai absolut menunjukkan besar kontribusi fitur tanpa membedakan arah pengaruhnya. Dengan demikian, dua client dapat memiliki feature importance yang serupa meskipun kontribusi positif dan negatif pada masing-masing sampel tidak identik.

Pada konteks evaluasi stabilitas interpretasi antar client, perbandingan nilai SHAP secara langsung kurang tepat karena setiap client memiliki distribusi fitur lokal yang berbeda di bawah kondisi Non-IID sehingga rentang nilai |𝜑𝑗| tidak setara antar client. Sebagai ilustrasi, suatu client yang memiliki proporsi transaksi fraud lebih tinggi dapat menghasilkan nilai SHAP yang lebih besar pada fitur tertentu dibandingkan client lain padahal urutan kepentingan fitur antar keduanya bisa jadi serupa. Untuk menetralkan perbedaan skala semacam ini, literatur interpretable machine learning merekomendasikan penggunaan metrik berbasis peringkat (rank-based metrics) yang bekerja pada urutan kepentingan fitur dan bukan pada nilai absolutnya (Doshi-Velez dan Kim, 2017).

Konsistensi peringkat fitur diukur menggunakan korelasi Spearman, yaitu korelasi antara peringkat dua vektor importance. Setiap client mengurutkan fitur berdasarkan 𝑔𝑐,𝑗 dari fitur paling berpengaruh hingga paling tidak berpengaruh, sehingga setiap fitur memperoleh nilai peringkat. Korelasi Spearman antara dua vektor peringkat tersebut dirumuskan pada Persamaan (2.9):

`<Persamaan 2.9>`

dengan 𝑑𝑗 sebagai selisih peringkat fitur ke-𝑗 antara dua client yang dibandingkan, dan 𝑑 sebagai jumlah fitur. Nilai positif yang mendekati satu menunjukkan urutan fitur yang semakin serupa, sedangkan nilai negatif menunjukkan kecenderungan urutan yang berlawanan. Nilai nol menunjukkan tidak adanya hubungan monoton yang terukur dan bukan bahwa seluruh fitur berbeda. Fitur dengan nilai yang sama memperoleh peringkat rata-rata. Pada vektor konstan, korelasi peringkat tidak terdefinisi dan tidak boleh dilaporkan sebagai kesepakatan sempurna.

Korelasi Spearman dipilih sebagai metrik stabilitas karena tiga alasan. Pertama, metrik ini menggunakan informasi peringkat penuh (full ranking information) dari seluruh fitur sehingga memberikan gambaran menyeluruh mengenai kesesuaian interpretasi pada semua tingkat kepentingan. Kedua, metrik ini scale-invariant sehingga tetap valid meskipun nilai SHAP berbeda antar client akibat heterogenitas data. Ketiga, korelasi peringkat memiliki interpretasi statistik dapat diuji signifikansinya sehingga klaim stabilitas dapat dipertanggungjawabkan secara statistik.

Selain korelasi tanpa bobot, penelitian ini menggunakan korelasi peringkat berbobot nilai agar perbedaan pada fitur dengan kontribusi lebih besar mendapat perhatian lebih tinggi. Ukuran ini dihitung sebagai korelasi Pearson berbobot atas kedua vektor peringkat. Bobotnya adalah satu vektor tunggal per sel pengukuran, yaitu rata-rata |𝜑| setiap fitur atas seluruh vektor importance client dan seed pada sel tersebut yang dinormalisasi agar berjumlah satu dan dipakai identik untuk setiap pasangan di dalam sel. Pembobotan ini melengkapi Spearman tanpa bobot yang memperlakukan seluruh posisi peringkat secara setara. Kedua ukuran tetap menilai kesamaan peringkat, bukan kesamaan angka SHAP secara langsung.

Kesamaan himpunan fitur terpenting dapat diukur melalui Jaccard similarity, yaitu ukuran irisan kedua himpunan dibagi ukuran gabungannya. Jaccard similarity pada top-𝐾 fitur dirumuskan pada Persamaan (2.10):

`<Persamaan 2.10>`

dengan 𝑇𝑎(𝐾) dan 𝑇𝑏(𝐾) masing-masing sebagai himpunan 𝐾 fitur teratas pada client 𝑎 dan client 𝑏. Berbeda dari korelasi Spearman yang menilai keseluruhan urutan, Jaccard similarity hanya menilai kesepakatan mengenai identitas himpunan fitur teratas tanpa memperhatikan urutan internal di antara fitur-fitur tersebut. Dua client yang sepakat bahwa lima fitur tertentu adalah yang paling penting akan memperoleh 𝐽5 = 1 meskipun urutan internal kelima fitur tersebut berbeda.

Kedua metrik dilaporkan bersama karena kombinasinya memungkinkan deteksi kondisi yang tidak dapat ditangkap oleh metrik tunggal. Dua client dapat memiliki korelasi Spearman yang tinggi namun Jaccard similarity yang rendah apabila sebagian besar fitur memiliki peringkat yang konsisten tetapi fitur-fitur teratasnya berbeda. Sebaliknya, dua client dapat memiliki Jaccard similarity yang tinggi namun korelasi Spearman yang rendah apabila himpunan fitur teratasnya identik tetapi urutan keseluruhan fiturnya berbeda. Namun, perbandingan antar dataset juga perlu memperhitungkan peluang kesamaan yang dipengaruhi jumlah fitur. Untuk itu, digunakan indeks Kuncheva (Kuncheva, 2007). Apabila dua client memilih 𝑘 fitur dari 𝑀 fitur dan memiliki 𝑟 fitur yang sama, indeks tersebut dinyatakan pada Persamaan (2.11):

`<Persamaan 2.11>`

Indeks ini memperhitungkan irisan yang dapat terjadi secara kebetulan. Nilai satu menunjukkan himpunan fitur yang identik, nilai nol menunjukkan kesamaan pada tingkat acuan kebetulan, dan nilai negatif menunjukkan irisan yang lebih rendah daripada acuan tersebut. Nilai satu tidak menyatakan bahwa urutan atau besarnya kontribusi fitur di dalam himpunan juga identik. Penelitian ini menggunakan lima fitur terpenting sebagai ringkasan utama, serta kurva terhadap 𝑘 untuk memeriksa ketergantungan hasil pada banyaknya fitur yang dipilih (Nogueira dkk, 2018).

Pada KernelSHAP tersampel, perbedaan vektor importance dapat muncul walaupun model, background, dan sampel yang dijelaskan tidak berubah. Oleh karena itu, kesepakatan antar client dibandingkan dengan kesepakatan dari pengulangan estimator di dalam client yang sama. Pengukuran dalam client menggambarkan variasi estimator pada konfigurasi yang sedang diperiksa. Perbedaan antar client yang belum dapat dibedakan dari variasi tersebut tidak langsung menunjukkan bahwa interpretasinya stabil.

Pengujian stabilitas perlu membedakan perbedaan antar client dari variasi estimator. Uji exchangeability membentuk distribusi pembanding melalui pertukaran pemasangan vektor importance di bawah hipotesis nol bahwa pemasangan tersebut dapat dipertukarkan. Seluruh pemasangan dapat dienumerasi ketika jumlah vektor cukup kecil. Enumerasi memberikan distribusi pengujian yang lengkap untuk statistik dan masukan tersebut tetapi interpretasinya tetap bergantung pada asumsi exchangeability.

Karena pengujian dilakukan pada beberapa konfigurasi, nilai p disesuaikan menggunakan prosedur Benjamini–Hochberg. Prosedur ini ditujukan untuk mengendalikan false discovery rate pada suatu keluarga pengujian di bawah asumsi yang sesuai sehingga keputusan tidak hanya bergantung pada nilai p mentah setiap konfigurasi (Benjamini dan Hochberg, 1995). Konfigurasi yang tidak menunjukkan perbedaan signifikan tetap tidak dapat disimpulkan memiliki interpretasi yang identik.

Perbandingan stabilitas antara IID dan Non-IID dilakukan secara berpasangan menggunakan uji Wilcoxon signed-rank satu arah. Pasangan dibentuk dari model, dataset, dan kondisi SMOTE yang sama sehingga arah selisih menunjukkan apakah stabilitas IID lebih tinggi daripada Non-IID. Uji ini menguji pola selisih pasangan dengan asumsi yang menyertainya dan hasilnya tidak menggantikan pengulangan pelatihan pada beberapa random seed.

### 2.2.10 Flower Framework

Flower (Friendly Federated Learning Research Framework) merupakan kerangka kerja open-source yang dikembangkan oleh Beutel dkk (2022) untuk simulasi dan implementasi sistem Federated Learning. Flower menyediakan abstraksi tingkat tinggi yang memungkinkan peneliti mengimplementasikan berbagai skema agregasi dan model dengan kompatibilitas terhadap backend populer seperti PyTorch, TensorFlow, dan scikit-learn. Penelitian ini menggunakan Flower sebagai infrastruktur simulasi FL dengan implementasi mengikuti baseline hfedxgboost pada repositori resmi Flower.

---
