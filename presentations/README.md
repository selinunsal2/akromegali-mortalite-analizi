# Sunum dosyaları

## Çalıştırılabilir JupyterLab sunumu

**[Akromegali_Canli_Sunum.ipynb](Akromegali_Canli_Sunum.ipynb)** — tam **12 slayt**. Gerçek veri ve depodaki MLflow özet tabloları kullanılır; grafikler ve kod hücreleri interaktiftir.

Sunum Notebook'unu açmadan önce depo kökünde `requirements.txt` dosyasındaki paketleri kurun. Projenin kök klasörü altında `presentations/` klasöründen Notebook'u açın. İlk gizli kurulum hücresi (Jupyter `Run All` sırasında) ana klasörü bulur; veri, resim ve özet CSV dosyalarını oradan okur.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

JupyterLab'da **Python (Akromegali)** veya paketleri yüklenmiş bir Python kernel'i seçin. Kodları çalıştırmak için normal Notebook görünümünde `Run → Run All Cells`; RISE kuruluysa `View → Render as Reveal Slideshow` kullanılabilir.

**Önemli:** Sunumda kullanılan `reports/*_from_mlflow.csv` dosyaları geçmişte kaydedilmiş deney değerleridir; yeni model eğitimi yapmazlar. Deneyleri yeniden üretmek için depo kökünde şu komutu kullanın:

```bash
python run_project.py --no-mlflow
```

## Sabit sunum (PDF)

Proje raporunun **12 sayfalık sabit PDF** sürümü sohbet içinde oluşturulmuştur ancak bu repoya ikili (binary) dosya olarak henüz eklenmemiştir. Bu klasöre `Akromegali_Mortalite_Bizim_Sonuclarimiz_Sunum.pdf` adıyla yüklenebilir. Kaydedilmemiş dosyayı burada varmış gibi göstermiyoruz.

Jupyter Notebook'tan kendi statik slayt HTML çıktısını oluşturmak için:

```bash
jupyter nbconvert --to slides presentations/Akromegali_Canli_Sunum.ipynb
```

Bu HTML, Chrome üzerinden PDF'e yazdırılabilir; slayt sayfa taşmaları PDF önizlemesinde ayrıca kontrol edilmelidir.
