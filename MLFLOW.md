# MLflow deney takibi

Bu projedeki `experiment_1.csv` dosyası MLflow arayüzünden dışa aktarılmış **6 run** içerir: DummyClassifier, LogisticRegression ve RandomForest modellerinin birer test ve birer RepeatedStratifiedKFold deneyi.

## Yerel MLflow'u açmak (Windows / Anaconda)

Özgün çalışmanın Python ortamı `akromegali` olarak görüntülenmiştir. Yerel bilgisayarda proje dizininde:

```bat
conda activate akromegali
cd C:\Users\W-10\Akromegali_Projesi
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Ardından tarayıcıda `http://127.0.0.1:5000` adresini açın. Bu komut **yalnızca yerel dizinde özgün `mlflow.db` mevcutsa** kayıtlı deneylere erişir. Sunucu açık kaldığı sürece arayüze erişilebilir.

Önemli: GitHub'a yüklenen `experiment_1.csv`, **MLflow sunucusunun kendisi veya `mlflow.db` dosyası değildir**. Başka bir bilgisayar yalnızca CSV ile aynı MLflow arayüzünü yeniden oluşturamaz. `mlartifacts/` klasörü de bu repository'de yer almıyor; dolayısıyla model artifact'lerinin erişilebilirliği buradan doğrulanamaz.

## Deneylerin yorumlanması

- `DummyClassifier`, `LogisticRegression`, `RandomForest`: ayrılmış test kümesi metrikleri.
- `DummyClassifier_RepeatedCV`, `LogisticRegression_RepeatedCV`, `RandomForest_RepeatedCV`: 5 kat × 5 tekrarlı CV ortalamaları ve standart sapmaları.
- PR-AUC alanı average precision hesabına dayanır.
- Vefat sınıfı pozitif kabul edilir (ham `Alive=0`).

## Bilinen hata

Ana Notebook'ta `mlflow.sklearn.log_model` ilk denemelerde skops tarafından güvenilmeyen `numpy.dtype` türü sebebiyle hata vermiştir. **Ardından yapılan kayıt işlemlerinde Logistic Regression ve Random Forest için başarı mesajı ve `models:/...` URI'leri gözlenmiştir.** Bu nedenle başlangıçtaki hatayı son durum olarak yorumlamamak gerekir. Yerel arayüz model kayıtlarının doğrulanması için asıl kaynak olmaya devam eder; GitHub üzerinde model artifact dosyaları paylaşılmamıştır. Güvenilmeyen türleri otomatik olarak izinli hale getirmeden önce kaynağı inceleyin; sadece güvenilir modelleri yükleyin.

## Yeniden çalıştırma

Yerel sunucunun arka uç veritabanını kullanarak `python run_project.py` yeni deneyler kaydedebilir. Eski run'lar üzerinde doğrudan düzenleme yapılmaz. Sadece yeni analiz çıktısı üretmek isteniyorsa:

```bash
python run_project.py --no-mlflow
```

**Veri gizliliği:** Sağlık verisi, kişisel kimlik bilgileri, yerel veritabanları ve ham artifact'ler yetkisiz biçimde herkese açık GitHub repository'sine eklenmemelidir.
