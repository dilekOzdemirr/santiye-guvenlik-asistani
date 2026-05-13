# Akıllı Şantiye Güvenlik Asistanı

Bu proje şantiye görüntülerinde baret ve yelek ihlallerini tespit edip kayıt altına alan uçtan uca bir demo sistemidir. Mevcut YOLO/OpenCV tespit koduna PostgreSQL veritabanı, Node.js API ve React + TailwindCSS yönetim paneli eklenmiştir.

## Proje Yapısı

```text
.
├── backend/                  # Express API
├── database/schema.sql        # PostgreSQL tablo şeması
├── frontend/                 # React + Tailwind yönetim paneli
├── api_client.py             # Python tespit kodunun API istemcisi
├── video_tespit.py           # Video üzerinden ihlal tespiti
├── tespit_sistemi_webcam.py  # Webcam üzerinden ihlal tespiti
├── egitim.py                 # YOLO eğitim kodu
└── docker-compose.yml        # PostgreSQL servisi
```

## Kişi 3 Kapsamı

- PostgreSQL tarafında `violations` tablosu oluşturuldu.
- Node.js/Express API ile ihlal kayıtları alınır, listelenir, durumları güncellenir.
- Python tespit kodu ihlal fotoğrafı kaydedince API'ye `POST /api/violations` isteği atar.
- React panelinden video linki veya video dosyası verilerek analiz başlatılır.
- Python analiz motoru videoyu arka planda işler; güvenli kişiler yeşil, ihlal olan kişiler kırmızı kutuyla işaretlenmiş çıktı videosu üretir.
- React panelinde video analiz durumu, çıktı videosu, toplam/açık/günlük ihlal sayıları, ihlal tablosu, fotoğraf önizleme ve durum güncelleme ekranı bulunur.

## Kurulum

Önce bağımlılıkları yükle:

```powershell
cd C:\Users\rabia\Desktop\projelerr\santiye-guvenlik-asistani
npm install
python -m pip install -r requirements.txt
```

PostgreSQL'i Docker ile başlat:

```powershell
docker compose up -d db
```

Docker yoksa ve bilgisayarda PostgreSQL kuruluysa, `postgres` kullanıcısıyla şu komutlar çalıştırılabilir:

```powershell
psql -U postgres -c "CREATE USER santiye WITH PASSWORD 'santiye123';"
psql -U postgres -c "CREATE DATABASE santiye_guvenlik OWNER santiye;"
psql -U postgres -d santiye_guvenlik -f database/schema.sql
```

Ortam dosyalarını hazırla:

```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

API ve paneli birlikte çalıştır:

```powershell
npm run dev
```

Varsayılan adresler:

- API: `http://localhost:4000`
- Panel: `http://localhost:5173`
- PostgreSQL: `localhost:5432`

## Panelden Video Analizi

Paneldeki `Video Analizi` alanından iki şekilde analiz başlatılabilir:

- YouTube/Shorts linki veya doğrudan `.mp4` gibi indirilebilir bir video linki girilir.
- Bilgisayardaki video dosyası seçilir.

Backend bu istek için `video_jobs` tablosunda bir analiz işi açar ve `analyze_video_job.py` dosyasını arka planda çalıştırır. Analiz sürerken panelden `İptal Et` ile işlem durdurulabilir. Analiz bittiğinde:

- Kırmızı kutu: baret veya yelek ihlali
- Yeşil kutu: baret ve yelek uygun
- Çıktı videosu: `backend/uploads/videos/` içinde tarayıcı uyumlu H.264 MP4
- Ses: Orijinal videoda ses varsa çıktı videosuna geri eklenir
- İhlal fotoğrafları: `ihlaller/`
- İhlal kayıtları: PostgreSQL `violations` tablosu

Not: YouTube linkleri için `yt-dlp` kullanılır. `python -m pip install -r requirements.txt` çalıştırılmadıysa YouTube linki analiz edilemez.

## Python Tespit Kodunu API'ye Bağlama

Backend çalışırken video veya webcam tespit dosyasını başlat:

```powershell
python video_tespit.py
```

Webcam için:

```powershell
python tespit_sistemi_webcam.py
```

Model dosyasını değiştirmek için `EKIPMAN_MODEL_PATH` kullanılabilir. Örneğin `best.pt` veya `last.pt` ile çalışmak için:

```powershell
$env:EKIPMAN_MODEL_PATH="best.pt"
python video_tespit.py
```

API adresi farklıysa:

```powershell
$env:SAFETY_API_URL="http://localhost:4000/api/violations"
python video_tespit.py
```

## API Sözleşmesi

İhlal ekleme:

```http
POST /api/violations
Content-Type: application/json
```

Örnek gövde:

```json
{
  "violationType": "Baret Yok, Yelek Yok",
  "photoPath": "ihlaller/ihlal_20260511_163000.jpg",
  "helmetDetected": false,
  "vestDetected": false,
  "dangerZone": false,
  "bbox": { "x1": 120, "y1": 64, "x2": 350, "y2": 430 },
  "source": "video"
}
```

Diğer endpointler:

- `GET /api/health`
- `GET /api/stats`
- `POST /api/video-jobs`
- `GET /api/video-jobs`
- `GET /api/video-jobs/:id/video`
- `GET /api/video-jobs/:id/preview`
- `GET /api/violations?status=open&type=Baret%20Yok`
- `PATCH /api/violations/:id/status`
- `GET /api/violations/:id/photo`

## Veritabanı Şeması

Ana tablolar: `video_jobs` ve `violations`

`video_jobs`:

| Alan | Açıklama |
| --- | --- |
| `id` | Video analiz numarası |
| `source` | Link veya yüklenen dosya yolu |
| `source_type` | `url` veya `upload` |
| `status` | `queued`, `running`, `completed`, `failed` |
| `processed_frames` | İşlenen kare sayısı |
| `total_frames` | Toplam kare sayısı |
| `output_video_path` | Kırmızı/yeşil kutulu çıktı videosu |
| `preview_frame_path` | Panel önizleme karesi |

`violations`:

| Alan | Açıklama |
| --- | --- |
| `id` | Otomatik ihlal numarası |
| `video_job_id` | Hangi video analizinden geldiği |
| `violation_type` | Baret yok, yelek yok, tehlikeli bölge vb. |
| `detected_at` | İhlal zamanı |
| `photo_path` | Kaydedilen fotoğraf yolu |
| `source` | `video`, `webcam`, `panel-demo` gibi kaynak bilgisi |
| `helmet_detected` | Baret var mı |
| `vest_detected` | Yelek var mı |
| `danger_zone` | Tehlikeli bölge ihlali var mı |
| `bbox` | Kişinin koordinatları |
| `status` | `open`, `resolved`, `false_alarm` |

## Sunum İçin Kısa Akış

1. `docker compose up -d db` ile PostgreSQL'i aç.
2. `npm run dev` ile API ve React panelini aç.
3. Panelde `Video Analizi` alanına video yükle veya `.mp4` linki gir.
4. Analiz durumunun `Sırada`, `Analiz Ediliyor`, `Tamamlandı` olarak değiştiğini göster.
5. Çıktı videosunda yeşil/kırmızı kutuları göster.
6. Panelde ihlali aç, fotoğrafı görüntüle ve durumu `Çöz` olarak güncelle.

## Notlar

- Büyük model dosyaları için `.gitignore` içinde `*.pt` vardır. Repo içinde mevcut `santiye_modeli.pt` kullanılabilir; yeni eğitimden çıkan `best.pt` veya `last.pt` dosyaları GitHub'a yüklenmeden yerelde kullanılmalıdır.
- Python tarafı ek paket gerektirmeden standart `urllib` ile API'ye istek gönderir.
- Fotoğraflar varsayılan olarak `ihlaller/` klasöründe saklanır ve backend bu klasörden güvenli şekilde servis eder.
