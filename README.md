# Akromegali Kohortunda Mortalite Tahmini

Galatasaray Üniversitesi Veri Bilimi Uygulamaları kapsamında, **607 hastalık** bir akromegali kohortunda mortalite sınıfının makine öğrenmesiyle incelenmesine yönelik **keşifsel / akademik** çalışma.

> **Önemli:** Bu model klinik karar destek aracı değildir. Bağımsız dış doğrulama yapılmamıştır; bazı ameliyat öncesi değişkenlerin seçilen tahmin anında gerçekten erişilebilir olduğu ayrıca doğrulanmalıdır.

## Araştırma sorusu

Tanı / ameliyat öncesi olduğu varsayılan beş değişken kullanılarak takip sonunda **vefat (death=1)** sınıfı ayırt edilebilir mi?

- Veri: 607 satır, 61 sütun; `Alive=0`: 51 vefat (%8,4); `Alive=1`: 556 hayatta (%91,6).
- Modelleme hedefi: `death = (Alive == 0).astype(int)`. **Pozitif sınıf vefattır.**
- Kullanılan özellikler: `Age_in_Diagnosis`, `Gender`, `Preop_GH`, `Preop_IGF_1`, `Preop_Tm_Size_MAX_mm`.
- `Age`, `Follow_Up_Duration_Month`, `Last_*`, ameliyat sonrası ve sonuç değişkenleri ana model özelliklerine dahil edilmez (veri sızıntısı riski).

## Bulgular

**Eğitim seti üzerinde** 5 katlı, 5 tekrarlı stratified cross-validation (`random_state=42`):

| Model | ROC-AUC (ortalama) | PR-AUC / AP (ortalama) | Vefat recall (ortalama) | Vefat F1 (ortalama) |
| --- | ---: | ---: | ---: | ---: |
| DummyClassifier | 0,5000 | 0,0845 | 0,0000 | 0,0000 |
| **Logistic Regression** | **0,8205** | **0,4043** | **0,7117** | **0,3223** |
| Random Forest | 0,7574 | 0,3020 | 0,4483 | 0,2835 |

Ayrılan test seti: **122 hasta (10 vefat)**; eğitim seti: **485 hasta (41 vefat)**.

| Model | Accuracy | ROC-AUC | PR-AUC / AP | Vefat precision | Vefat recall | Vefat F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DummyClassifier | 0,9180 | 0,5000 | 0,0820 | 0,0000 | 0,0000 | 0,0000 |
| **Logistic Regression** | 0,7049 | **0,8049** | **0,3710** | 0,1750 | **0,7000** | **0,2800** |
| Random Forest | 0,7541 | 0,6683 | 0,3687 | 0,1667 | 0,5000 | 0,2500 |

**Yorum:** Tek başına accuracy yanıltıcıdır: Dummy model vefatları hiç yakalamadan %91,8 doğruluk alır. Lojistik regresyon, bu deneylerde vefat sınıfı için en yüksek çapraz doğrulama ROC-AUC, AP ve recall ortalamalarını sağlamıştır. Testteki yalnızca 10 vefat ve sınıf dengesizliği sonuçların belirsizliğini artırır; mevcut test kümesi geliştirme sürecinde daha önce incelendiğinden tam anlamıyla dokunulmamış bir test değerlendirmesi değildir. Ağırlıklandırılmış modelin verdiği skorlar **kalibre klinik risk olasılıkları** olarak yorumlanmamalıdır.

## Dosya yapısı

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Akromegali_Modelleme.ipynb   # Etkileşimli çalışma (orijinal)
├── run_project.py               # Tek komutla tekrar çalıştırılabilir script
├── akromegali_alistirma.csv     # Akademik çalışma verisi
├── experiment_1.csv             # MLflow deney özetinin arşiv kaydı
└── docs/
    └── reproducibility.md      # Yöntem, veri sözleşmesi ve tekrar üretme notları
```

**Not:** Ham hasta verisini içeren CSV hâlihazırda bu herkese açık depodadır. Yayın ve paylaşım için veri kullanım/anonimleştirme iznini ayrıca kontrol edin; dosya geçmişi herkese açık olabilir. Deney takibinin yerel veritabanı, model artefaktları ve büyük `results/` çıktıları depoya dahil edilmez.

## Kurulum ve çalıştırma

Python 3.11+ ve sanal ortam önerilir; bu çalışma Windows üzerinde `akromegali` conda ortamında yürütüldü.

```bash
python -m pip install -r requirements.txt
python run_project.py --no-mlflow
```

Komut, varsayılan `akromegali_alistirma.csv` dosyasını okuyarak `results/` dizinine metrik tablolarını, karmaşıklık matrislerini, hedef dağılımını, CV karşılaştırma grafiğini, lojistik regresyon katsayılarını, tek bir yanlış pozitif örnek açıklamasını ve çalıştırma meta verisini yazar.

Özel yollarla çalıştırma:

```bash
python run_project.py --data akromegali_alistirma.csv --output results --no-mlflow
```

MLflow sunucusu **zaten çalışıyorsa**, kayıt eklemek için `--no-mlflow` bayrağını kaldırabilirsiniz. Varsayılan takip adresi `http://127.0.0.1:5000` olup yeni run'lar oluşur; geçmiş run'ları güncellemez.

Jupyter üzerinden etkileşimli analiz için `Akromegali_Modelleme.ipynb` dosyasını açın ve **Python (Akromegali)** kernel'ini seçin. Notebook, önceki keşifsel çalışmadan kalan bazı etkileşimli hücreler içerir; **tekrar üretim için esas giriş noktası `run_project.py` dosyasıdır.**

## Yeniden üretilebilirlik

- Ayırma: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Yalnızca eğitim setinde: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`.
- Eksik değer işlemleri ve kodlama: `Pipeline` / `ColumnTransformer` içinde, her eğitim katmanında yeniden öğrenilir.
- Sayısal özellikler: medyan ile doldurma + eksiklik göstergesi; LR için `StandardScaler`. Cinsiyet: en sık değerle doldurma + `OneHotEncoder`.
- Modeller: çoğunluk sınıfı Dummy; dengeli sınıf ağırlıklı lojistik regresyon; 200 ağaçlı, maksimum derinliği 5 olan dengeli Random Forest.
- Tekrar çalıştırma Python ve kütüphane sürümleriyle ilişkili olabilir. Deney sırasında: scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.4.6, MLflow 3.17.2.
- Kayıtlı `experiment_1.csv`, geçmiş bir MLflow dışa aktarımıdır; **scriptin güncel çıktılarının yerine geçmez**.

Ayrıntılı açıklama: [Tekrar üretilebilirlik ve sınırlılıklar](docs/reproducibility.md).

## Akademik kullanım / sınırlılıklar

Veri seti küçük ve dengesizdir; sonuçlar klinik geçerliliği kanıtlamaz. Eksiklik göstergeleri hastanın tedavi / takip sürecine ilişkin dolaylı bilgi taşıyabilir. Model açıklamalarındaki katsayılar nedensel etki anlamına gelmez. Yeniden değerlendirme için bağımsız kohort, tahmin zamanı tanımı, kalibrasyon ve önceden belirlenmiş test protokolü gerekir.
