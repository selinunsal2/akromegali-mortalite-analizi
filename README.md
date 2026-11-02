# Akromegali Hastalarında Mortalite Tahmini

> Klinik özelliklerden mortalite durumunu tahmin etmeye yönelik makine öğrenmesi ön çalışması. Bu repository eğitim/araştırma amaçlıdır; klinik karar aracı değildir.

## Proje özeti

Bu projede akromegali hastalarına ait veri seti üzerinde keşifsel analiz, veri kalite kontrolleri, sınıflandırma, test değerlendirmesi ve tekrarlı çapraz doğrulama yapılmaktadır. **Hedef değişken:** `Alive` (0 = vefat, 1 = hayatta). Modelleme sırasında pozitif sınıf **vefat** olacak şekilde `(Alive == 0)` dönüştürülür; dolayısıyla precision, recall, F1, ROC-AUC ve PR-AUC yorumlarında pozitif olay vefattır.

Kullanılan girdiler: `Age_in_Diagnosis`, `Gender`, `Preop_GH`, `Preop_IGF_1`, `Preop_Tm_Size_MAX_mm`. Ana çalışma, **DummyClassifier**, **Logistic Regression** ve **Random Forest** modellerini karşılaştırır. Sayısal özellikler için eksik değer doldurma, lojistik regresyonda ölçekleme ve kategorik değişkenlerde OneHotEncoder kullanılır. Ön işleme adımları model pipeline'larının içindedir.

## Kaydedilmiş sonuçlar

Aşağıdaki değerler repository'deki **`experiment_1.csv` MLflow dışa aktarımından** alınmıştır; yeniden hesaplanmış veya değiştirilmiş değildir.

### Ayrılmış test kümesi (%20)

| Model | Accuracy | Precision (vefat) | Recall (vefat) | F1 (vefat) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.9180 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0820 |
| Logistic Regression | 0.7049 | 0.1750 | 0.7000 | 0.2800 | 0.8049 | 0.3710 |
| Random Forest | 0.7541 | 0.1667 | 0.5000 | 0.2500 | 0.6683 | 0.3687 |

### Tekrarlı çapraz doğrulama (eğitim verisinde 5 kat × 5 tekrar)

| Model | Ortalama ROC-AUC ± SS | Ortalama PR-AUC ± SS | Ortalama Recall (vefat) ± SS |
|---|---:|---:|---:|
| DummyClassifier | 0.5000 ± 0.0000 | 0.0845 ± 0.0041 | 0.0000 ± 0.0000 |
| Logistic Regression | 0.8205 ± 0.0717 | 0.4043 ± 0.1360 | 0.7117 ± 0.1703 |
| Random Forest | 0.7574 ± 0.0685 | 0.3020 ± 0.0925 | 0.4483 ± 0.1377 |

**Yorum:** Sınıf dengesizliği nedeniyle yüksek accuracy tek başına yeterli bir başarı göstergesi değildir. DummyClassifier yüksek accuracy değerine rağmen vefat sınıfını yakalayamamıştır. Bu kayıtlar için Logistic Regression, test ROC-AUC ve vefat recall değerlerinde Random Forest'tan yüksek sonuç vermiştir. Bunlar yalnızca bu veri bölünmesine/deney düzenine aittir; klinik geçerlilik veya genelleme garantisi değildir.

## Repository dosyaları

| Yol | Açıklama |
|---|---|
| `Akromegali_Modelleme.ipynb` | Ana Jupyter Notebook, analiz ve kaydedilmiş çıktılar |
| `akromegali_alistirma.csv` | Projede kullanılan veri dosyası |
| `run_project.py` | Komut satırından yeniden çalıştırma ve rapor üretimi |
| `experiment_1.csv` | MLflow dışa aktarılmış 6 run: 3 test + 3 çapraz doğrulama |
| `requirements.txt` | Gerekli Python paketleri |
| `MLFLOW.md` | MLflow'un başlatılması, deneylerin anlamı ve sınırlamalar |
| `.gitignore` | Yerel / geçici dosyalar için ignore kuralları |

## Kurulum ve yeniden çalıştırma

Python ortamında (örneğin Anaconda Prompt'ta):

```bash
conda create -n akromegali-repro python=3.11 -y
conda activate akromegali-repro
python -m pip install -r requirements.txt
```

Komut satırı analizi için (MLflow sunucusu olmadan):

```bash
python run_project.py --no-mlflow
```

Bu komut `results/` klasörüne analiz çıktıları üretir. Notebook'u incelemek için `jupyter notebook` komutunu çalıştırıp `Akromegali_Modelleme.ipynb` dosyasını açın. Notebook'un bazı hücreleri önceden oluşturulmuş çıktılar veya çalışan yerel MLflow sunucusu gerektirebilir; bütün hücrelerin temiz ortamda baştan sona çalıştığı henüz doğrulanmamıştır.

MLflow kullanmak için bkz. [MLFLOW.md](MLFLOW.md). **`experiment_1.csv` bir sonuç dışa aktarımıdır; sunucu veritabanı ya da eğitilmiş model dosyası değildir.**

## Yöntem ve yeniden üretilebilirlik notları

- Betikte ayrılmış test kümesi oranı %20, rastgele tohum 42'dir; sınıf oranları korunarak bölme yapılır.
- Tekrarlı çapraz doğrulama eğitim bölümünde 5 kat × 5 tekrar üzerinden yapılır. Ön işleme pipeline içinde yeniden öğrenilir.
- MLflow CSV dosyası önceden kaydedilmiş sonuçları gösterir. Yeni çalıştırmalarda kütüphane sürümü veya veriler değişirse sayılar farklılaşabilir.
- Notebook çıktısında MLflow'un `skops` serileştirmesi sırasında `numpy.dtype` için **untrusted types** hatası görülmektedir. Bu nedenle bütün eğitilmiş model artifact'lerinin başarılı biçimde kaydedildiği varsayılmamalıdır; mevcut run metrikleri ayrı olarak mevcuttur.
- Repo'nun mevcut haliyle sonuçların birebir yeniden üretildiği bağımsız bir ortamda doğrulanmamıştır.

## Veri güvenliği ve etik

Sağlık verileri hassastır. Repository herkese açıksa kaynak veri, dışa aktarılmış deneyler, kaydedilmiş Notebook çıktıları ve model artifact'lerinde kimlik belirleyen bilgiler bulunmadığı ve paylaşım yetkisi olduğu ayrıca doğrulanmalıdır. Bu modeller hasta bazında tıbbi karar vermek için kullanılmamalıdır.

## Kapsam

**Akademik ön çalışma · Python · scikit-learn · Jupyter Notebook · MLflow**
