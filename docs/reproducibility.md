# Tekrar Üretilebilirlik ve Metodoloji Notları

## Çalışma akışı

1. `akromegali_alistirma.csv` okunur; hedef `Alive=0` ise `death=1` olacak şekilde tanımlanır.
2. Betik temel veri sözleşmesi denetimleri ve bilinçli bozulma testi yapar. Kayıt: `results/data_contract.csv` ve `results/corruption_test.json`.
3. Sadece beş öngördürücü değişken seçilir: `Age_in_Diagnosis`, `Gender`, `Preop_GH`, `Preop_IGF_1`, `Preop_Tm_Size_MAX_mm`.
4. Ön işleme öncesinde eğitim/test ayrımı yapılır; sınıf oranı korunur ve `random_state=42` kullanılır.
5. `SimpleImputer`, eksiklik göstergeleri, ölçekleme/kodlama model `Pipeline` içinde kalır. CV ölçümleri yalnızca eğitim setinden hesaplanır.
6. Dummy, lojistik regresyon ve Random Forest karşılaştırılır; test ölçümleri betik tarafından betimleyici olarak raporlanır.
7. Lojistik regresyonun küresel katsayıları ve bir yanlış pozitif vakanın logit katkıları çıktı olarak kaydedilir.

## Veri sözleşmesi

Scriptte üç kural kontrol edilir:
- `Alive` boş olmayacak ve `{0,1}` kümesinde bulunacak.
- İlgili değerleri dolu satırlarda `Age_in_Diagnosis <= Age_at_Operation <= Age`.
- İlgili değerleri dolu satırlarda `TSS_VAR_YOK == "YOK"` iff `Remission_3_Month == "NOT_OPERATED"`.

Kontrol edilemeyen (boş) satırlar ayrıca sayılır; bir satırın ameliyat yaşı özellikle değiştirilerek kuralın ihlali yakaladığı test edilir. Bu kurallar veri sözleşmesinin tüm olası hata türlerini kapsadığı anlamına gelmez.

## Hedef ve metriklerin yorumu

`death=1` pozitif sınıftır. `pr_auc` anahtarının betikteki hesabı `average_precision_score` (average precision / AP) ile yapılır. ROC-AUC, AP, vefat precision/recall/F1 raporlanır. Çoğunluk sınıfı Dummy'nin accuracy değeri yüksek görünse de vefat recall değeri sıfırdır.

## Sınırlılıklar

- `Preop_*` alanlarının hepsinin belirlenen tanı / ameliyat öncesi tahmin noktasında bilindiği kaynak belgelerle doğrulanmış değildir; **olası zaman sızıntısı** kalır.
- Takip sonuna ilişkin `Age`, `Follow_Up_Duration_Month`, `Last_*`, `Postop_*` benzeri alanlar modele alınmaz.
- Test setinde yalnızca 10 vefat bulunur. Eğitim geliştirme sırasında test sonuçları daha önce incelendiği için raporlanan test değerleri tamamen bağımsız doğrulama sayılmamalıdır.
- `class_weight="balanced"` nedeniyle lojistik regresyon skorları doğrudan kalibre mortalite olasılığı değildir.
- Katsayılar, eksiklik göstergeleri ve tekil hasta açıklamaları nedensellik veya klinik eylem önerisi ifade etmez.
- Tekrarlı CV katmanları bağımsız gözlemler değildir; katmanlar arası standart sapma güven aralığı olarak sunulmamalıdır.

## Tek komutla çalıştırma

```bash
python -m pip install -r requirements.txt
python run_project.py --no-mlflow
```

Scriptin oluşturacağı `results/` çıktıları sürüm kontrolünden hariç tutulur. Önceki MLflow CSV arşivi karşılaştırma kaydıdır; çalıştırmanın gerçek sonuçları için yeni oluşan `results/test_results.csv` ve `results/cross_validation_results.csv` dosyalarına bakın.
