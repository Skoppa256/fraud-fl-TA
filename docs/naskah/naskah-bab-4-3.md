# 4.3 Interpretabilitas Model

## 4.3.1 Perbandingan Konsistensi Interpretasi Antar Model

Hasil penelitian menunjukkan bahwa model parametrik LR dan SVM cenderung menghasilkan interpretasi yang lebih konsisten antar client dibandingkan model lainnya pada kumpulan konfigurasi yang dievaluasi. Namun, konsistensi tersebut tidak membentuk urutan yang selalu sama pada seluruh dataset. Model berbasis tree maupun deep learning juga dapat menghasilkan kesamaan interpretasi yang tinggi tetapi memiliki variasi yang lebih besar pada kondisi tertentu. Tabel 4.12 menyajikan median dan rentang korelasi peringkat berbobot antar client pada pengukuran utama.

**Tabel 4.12. Korelasi peringkat berbobot antar client pada pengukuran utama**

| Model | Median | Minimum | Maksimum |
|---|---|---|---|
| LR | 0,9845 | 0,9064 | 0,9970 |
| SVM | 0,9895 | 0,8980 | 0,9989 |
| GBM | 0,9774 | 0,8014 | 0,9960 |
| FFD | 0,9634 | 0,7979 | 0,9924 |
| BERT | 0,9517 | 0,8682 | 0,9949 |
| FedXGBllr | 0,9576 | 0,6650 | 0,9859 |

Model LR dan SVM memperoleh median korelasi sebesar 0,9845 dan 0,9895, diikuti GBM sebesar 0,9774. Adapun FFD, BERT, dan FedXGBllr memperoleh median sebesar 0,9634, 0,9517, dan 0,9576. Nilai tersebut menunjukkan bahwa secara umum urutan kepentingan fitur masih memiliki kesamaan yang tinggi antar client. Namun, median yang tinggi belum menggambarkan seluruh kondisi sebagaimana terlihat pada FedXGBllr yang memiliki rentang dari 0,6650 hingga 0,9859. Maka dari itu, karakteristik interpretabilitas perlu dilihat bersama dengan variasinya antar dataset dan kondisi partisi.

Kesamaan lima fitur terpenting menunjukkan pola yang searah pada kelompok LR, SVM, dan GBM. Rata-rata indeks Kuncheva mencapai 0,9433 pada SVM, 0,9112 pada LR, dan 0,8219 pada GBM. Meskipun model parametrik memiliki nilai lebih tinggi secara keseluruhan, perbandingan per dataset menunjukkan bahwa keunggulan tersebut tidak berlaku pada seluruh kondisi.

**Tabel 4.13. Rerata indeks Kuncheva lima fitur terpenting per dataset**

| Dataset | LR | SVM | GBM |
|---|---|---|---|
| BAF | 0,956 | 0,927 | 0,853 |
| ULB | 1,000 | 0,964 | 0,742 |
| PaySim | 0,789 | 0,935 | 0,878 |

Pada ULB, LR menghasilkan indeks Kuncheva sebesar 1,000 dibandingkan GBM sebesar 0,742. Sebaliknya, pada PaySim, GBM memperoleh nilai 0,878 yang lebih tinggi daripada LR sebesar 0,789. Hasil ini menunjukkan bahwa model linear tidak selalu menghasilkan kesamaan fitur terpenting yang lebih tinggi daripada model tree. Nilai 1,000 pada LR di ULB menunjukkan bahwa himpunan lima fitur terpenting sama antar client tetapi tidak berarti urutan maupun besaran atribusi seluruh fiturnya identik.

Adapun pengaruh kondisi partisi menunjukkan bahwa interpretasi cenderung lebih konsisten pada kondisi IID. Pada FFD, BERT, dan FedXGBllr, korelasi peringkat berbobot menurun pada kondisi Non-IID dalam 11 dari 15 pasangan yang dapat dibandingkan. Pola tersebut paling konsisten pada PaySim dimana seluruh enam pasangan menunjukkan penurunan. Namun, arah yang sama tidak ditemukan pada seluruh model dan dataset sehingga Non-IID tidak dipahami sebagai penyebab penurunan yang bersifat mutlak.

Secara keseluruhan, model parametrik cenderung memiliki konsistensi feature importance yang tinggi sedangkan model tree, deep learning, dan tree ensemble menunjukkan variasi yang lebih bergantung pada dataset dan kondisi partisi. Kondisi Non-IID cenderung menurunkan konsistensi tersebut tetapi besarnya pengaruh berbeda antar model. Perbedaan ini tidak dapat dikaitkan hanya dengan paradigma agregasi karena arsitektur model dan metode penjelasannya juga berbeda. Selain itu, konsistensi interpretasi merupakan karakteristik yang berbeda dari performa deteksi sehingga model dengan AUPRC tinggi belum tentu menghasilkan penjelasan yang sama pada seluruh client.

## 4.3.2 Pengaruh Heterogenitas Distribusi Data dan SMOTE terhadap Interpretasi

Pengaruh heterogenitas distribusi data pada FFD, BERT, dan FedXGBllr menunjukkan penurunan rata-rata korelasi peringkat berbobot sebesar 0,054 dari kondisi IID ke Non-IID pada pengukuran utama. Penurunan terjadi pada 11 dari 15 pasangan, dengan nilai p sebesar 0,0042 berdasarkan uji Wilcoxon berpasangan satu sisi. Hasil ini menunjukkan kecenderungan penurunan kesamaan interpretasi meskipun arah perubahan pada setiap konfigurasi tidak selalu sama.

Pengaruh paling konsisten ditemukan pada PaySim, dimana seluruh enam pasangan menunjukkan korelasi yang lebih rendah pada kondisi Non-IID. Namun, besar penurunannya berbeda antar model dan kondisi SMOTE sebagaimana disajikan pada Tabel 4.14.

**Tabel 4.14. Korelasi peringkat berbobot antar client pada PaySim menggunakan KernelSHAP tersampel**

| Model | Kondisi SMOTE | IID | Non-IID |
|---|---|---|---|
| FedXGBllr | Tanpa SMOTE | 0,986 | 0,822 |
| FedXGBllr | Dengan SMOTE | 0,948 | 0,665 |
| FFD | Tanpa SMOTE | 0,973 | 0,798 |
| FFD | Dengan SMOTE | 0,931 | 0,886 |
| BERT | Tanpa SMOTE | 0,983 | 0,941 |
| BERT | Dengan SMOTE | 0,972 | 0,952 |

Penurunan terbesar terjadi pada FedXGBllr dengan SMOTE, yaitu dari 0,948 pada kondisi IID menjadi 0,665 pada kondisi Non-IID. Tanpa SMOTE, nilainya juga menurun dari 0,986 menjadi 0,822. Adapun FFD tanpa SMOTE mengalami penurunan dari 0,973 menjadi 0,798 sedangkan BERT menunjukkan perubahan yang lebih kecil pada kedua kondisi SMOTE. Hasil ini menunjukkan bahwa ketiga model memiliki sensitivitas interpretasi yang berbeda terhadap heterogenitas data meskipun seluruhnya menggunakan metode penjelasan yang sama.

Perubahan tersebut tidak selalu searah dengan perubahan performa deteksi. Pada PaySim tanpa SMOTE, AUPRC FedXGBllr tetap sekitar 0,996 pada kondisi IID maupun Non-IID sedangkan kesamaan interpretasinya menurun. Dengan demikian, kemampuan model mempertahankan performa deteksi tidak menjamin bahwa peringkat fitur yang dianggap penting akan tetap konsisten antar client. Dalam pengukuran ini, penurunan konsistensi menunjukkan bahwa penjelasan model global menjadi lebih sensitif terhadap perbedaan background lokal.

Pengaruh Non-IID juga ditemukan pada model dengan explainer deterministik. Pada GBM di PaySim tanpa SMOTE, indeks Kuncheva menurun dari 1,000 pada kondisi IID menjadi 0,7725 pada kondisi Non-IID. Pada ULB tanpa SMOTE, nilainya menurun dari 0,688 menjadi 0,544. Namun, LR di ULB tetap memiliki indeks Kuncheva 1,000 pada kedua kondisi. Maka dari itu, heterogenitas tidak selalu mengubah himpunan lima fitur terpenting dan pengaruhnya tetap bergantung pada model serta dataset.

Penanganan class imbalance dengan SMOTE turut memengaruhi konsistensi interpretasi. Pada kelompok LR, SVM, dan GBM, rata-rata kenaikan indeks Kuncheva setelah SMOTE sebesar 0,048 pada enam pasangan IID dan 0,085 pada sembilan pasangan Non-IID. Salah satu peningkatan terbesar terjadi pada GBM di ULB kondisi Non-IID, yaitu dari 0,544 menjadi 0,904. Adapun LR dan SVM di BAF kondisi Non-IID meningkat dari 0,868 menjadi 1,000.

**Tabel 4.15. Perubahan konsistensi dan fitur penting setelah SMOTE pada beberapa konfigurasi Non-IID**

| Dataset dan model | Kuncheva tanpa SMOTE | Kuncheva dengan SMOTE | Perubahan fitur penting |
|---|---|---|---|
| ULB - GBM | 0,544 | 0,904 | V7 keluar dari lima fitur terpenting, sedangkan V3 masuk pada peringkat ketiga seluruh client |
| BAF - LR | 0,868 | 1,000 | Fitur dominan berubah dari prev_address_months_count_missing menjadi housing_status_BB |
| BAF - SVM | 0,868 | 1,000 | Lima fitur terpenting setelah SMOTE terdiri dari empat indikator housing_status dan has_other_cards |

Peningkatan konsistensi tersebut disertai perubahan fitur yang dianggap penting. Pada GBM di ULB, V7 yang sebelumnya berada pada peringkat kedua seluruh client tidak lagi termasuk dalam lima fitur terpenting setelah SMOTE. Sebaliknya, V3 masuk pada peringkat ketiga seluruh client. Pada LR di BAF, fitur dominan berubah dari indikator ketidaktersediaan riwayat alamat menjadi kategori status tempat tinggal. Hasil ini menunjukkan bahwa SMOTE dapat meningkatkan kesamaan interpretasi dengan membentuk kesepakatan terhadap kelompok fitur yang berbeda.

Namun, pengaruh positif terhadap konsistensi tidak ditemukan pada seluruh model. Pada PaySim kondisi Non-IID, korelasi FFD meningkat dari 0,798 menjadi 0,886 setelah SMOTE dan BERT meningkat dari 0,941 menjadi 0,952. Sebaliknya, FedXGBllr menurun dari 0,822 menjadi 0,665. Dengan demikian, pengaruh SMOTE terhadap interpretasi bergantung pada interaksinya dengan model dan distribusi data sebagaimana pengaruhnya terhadap performa prediksi.

Peningkatan konsistensi juga belum tentu menunjukkan peningkatan kualitas deteksi. Pada LR di BAF kondisi Non-IID, indeks Kuncheva meningkat dari 0,868 menjadi 1,000 sedangkan AUPRC menurun dari sekitar 0,139 menjadi 0,097. Maka dari itu, kesepakatan antar client mengenai fitur terpenting perlu dibaca bersama dengan isi penjelasan dan performa model. Perubahan setelah SMOTE mencakup model yang dilatih serta background yang digunakan sehingga belum dapat dikaitkan hanya dengan salah satu komponen tersebut.

## 4.3.3 Keterandalan dan Batas Interpretasi Hasil SHAP

Konsistensi interpretasi antar client perlu dibedakan dari konsistensi pengukuran SHAP itu sendiri. LR, SVM, dan GBM menghasilkan atribusi yang sama ketika perhitungan diulang dengan masukan tetap. Adapun FFD, BERT, dan FedXGBllr menggunakan KernelSHAP tersampel sehingga perbedaan hasil dapat berasal dari distribusi data antar client maupun variasi sampling. Pengulangan dua seed pada client yang sama digunakan untuk mengukur besarnya variasi pengukuran tersebut.

**Tabel 4.16. Kesepakatan pengulangan dan kesepakatan antar client pada KernelSHAP tersampel**

| Model | Median kesepakatan dua seed dalam client | Median kesepakatan antar client | Konfigurasi dengan perbedaan terdeteksi |
|---|---|---|---|
| FFD | 0,9997 | 0,9634 | 11 dari 11 |
| BERT | 0,9993 | 0,9517 | 10 dari 11 |
| FedXGBllr | 0,9988 | 0,9576 | 8 dari 11 |

Kesepakatan dua pengulangan dalam client mendekati 1 pada ketiga model sedangkan kesepakatan antar client lebih rendah. Hasil ini menunjukkan bahwa pengulangan pada masukan yang sama menghasilkan peringkat yang lebih serupa dibandingkan pengukuran menggunakan background client yang berbeda. Berdasarkan prosedur pengujian dan koreksi pengujian berganda yang digunakan, 29 dari 33 konfigurasi menghasilkan indikasi perbedaan antar client.

Kesamaan peringkat yang tinggi dengan demikian masih dapat disertai perbedaan yang terdeteksi secara statistik. Sebaliknya, empat konfigurasi yang tidak menunjukkan perbedaan signifikan belum memberikan bukti yang cukup untuk membedakan variasi antar client dari variasi estimator. Hasil tersebut tidak langsung berarti bahwa interpretasinya identik atau stabil.

Kesamaan himpunan fitur, urutan fitur, dan besaran atribusi juga memiliki arti yang berbeda. Indeks Kuncheva sebesar 1,000 menunjukkan kesamaan himpunan fitur terpilih tetapi tidak memastikan bahwa urutan maupun kontribusi setiap fitur sama. Demikian pula, korelasi peringkat yang mendekati 1 menunjukkan urutan yang hampir seragam tanpa mengharuskan besaran SHAP identik. Oleh karena itu, hasil konsistensi pada penelitian ini terutama menjawab kesepakatan mengenai fitur penting dan peringkatnya.

Secara keseluruhan, hasil pengukuran mendukung pembacaan bahwa konsistensi interpretasi bergantung pada model, dataset, dan kondisi distribusi data. Namun, tingginya satu nilai kesamaan belum cukup untuk menyatakan kualitas explainability secara menyeluruh. Penilaian perlu mempertimbangkan fitur yang dianggap penting, perubahan peringkatnya, serta keterandalan pengukurannya. Temuan ini berlaku pada model akhir yang dievaluasi dalam penelitian dan belum mengukur kestabilan interpretasi lintas pengulangan pelatihan.
