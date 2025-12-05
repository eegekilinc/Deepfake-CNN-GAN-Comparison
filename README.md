# 🎭 Deepfake Tespiti ve Üretimi: Ön İşleme ve GAN Karşılaştırmalı Analiz

Bu deepfake tespitinde **ön işleme tekniklerinin etkisini** inceleyen ve **LightGAN** mimarisi ile sahte yüzler üreten karşılaştırmalı bir yapay zekâ projesidir.

Proje, hem deepfake tespitindeki sınıflandırma başarısını optimize etmeyi hem de deepfake üretim süreçlerini hafif bir GAN mimarisi üzerinden anlamayı amaçlamaktadır.

---

## ✨ Proje Özeti ve Ana Bulgular

Bu projede iki ana çalışma yürütülmüştür:

### 1. Deepfake Sınıflandırma (Detection)

Görüntü bütünlüğünün tespiti için temel bir **CNN Sınıflandırıcı** kullanılmış ve ön işleme tekniklerinin etkisi karşılaştırılmıştır.

 Gaussian Blur, yüz hatlarının keskinliğini azaltarak, modelin **sahte/gerçek ayrımındaki** kararlılığını artırmıştır.

### 2. LightGAN ile Deepfake Üretimi (Synthesis)

Düşük kapasiteli bir GAN mimarisi olan **LightGAN**, 10 epoch boyunca eğitilerek çözünürlüğünde deepfake yüz görüntüleri üretilmiştir.

* **Mimari:** Basit Evrişimsel Katmanlar ve `Conv2DTranspose` katmanlarından oluşur.
* **Sonuç:** Yüzün genel yapısı öğrenilmiş, ancak 10 epoch ve hafif mimari nedeniyle çıktılar bulanık ve kararsızdır.

---

## 🛠️ Kurulum ve Kullanım

### 1. Veri Seti

Proje, Kaggle'daki **Deepfake Dataset**'i (`aryanasingh16/deepfake-dataset`) kullanmaktadır. Çalıştırmak için bu veri setinin yerel/Colab ortamında `real_vs_fake/real-vs-fake` dizinine indirilmesi gereklidir.

| Klasör | Fake Görüntü | Real Görüntü |
| :---:  | :---:        | :---:        |
| `train`| ~50.960      | ~51.081      |
| `valid`| 10.000       | 10.000       |
| `test` | 10.000       | 10.000       |

### 2. Bağımlılıklar

Gerekli kütüphaneler (Jupyter Notebook'tan alınmıştır):
```bash
pip install tensorflow matplotlib pandas tqdm opencv-python --quiet