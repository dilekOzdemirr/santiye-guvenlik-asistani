# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:43:49 2026

@author: zeren
"""
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# İhlal tespitlerinin kaydedileceği dizinin kontrolü ve oluşturulması
if not os.path.exists("ihlaller"):
    os.makedirs("ihlaller")

print("Yapay zeka modelleri belleğe yükleniyor...")

# 1. Aşama: İnsan tespiti için önceden eğitilmiş temel model (COCO veri seti)
insan_modeli = YOLO("yolov8n.pt") 

# 2. Aşama: Baret ve yelek tespiti için özel eğitilmiş model
ekipman_modeli = YOLO("santiye_modeli.pt")

print("Modeller hazır. Video analizi başlatılıyor...")

# Video kaynağının tanımlanması
kamera = cv2.VideoCapture("videoplayback.mp4") 

# Ardışık ihlal kayıtlarını önlemek için zaman damgası
son_fotograf_zamani = 0  

while True:
    ret, frame = kamera.read()
    
    # Video akışı bittiğinde veya kesildiğinde döngüyü sonlandır
    if not ret:
        print("Video analizi tamamlandı veya görüntü alınamıyor.")
        break
        
    # İşlem yükünü azaltmak ve performansı artırmak için çözünürlük düşürme
    frame = cv2.resize(frame, (640, 480))
        
    # 1. Aşama: Görüntü üzerindeki kişilerin (Sınıf 0) tespiti
    insan_sonuclar = insan_modeli.predict(source=frame, classes=[0], verbose=False) 
    
    su_an = time.time()
    
    for sonuc in insan_sonuclar:
        kutu_koordinatlari = sonuc.boxes.xyxy 
        
        for kutu in kutu_koordinatlari:
            x1, y1, x2, y2 = map(int, kutu)
            
            # Tespit edilen kişinin koordinatlarına göre İlgi Alanı (Region of Interest - ROI) çıkarımı
            insan_bolgesi = frame[y1:y2, x1:x2]
            
            # Hatalı veya işlenemeyecek kadar küçük tespitleri yoksay
            if insan_bolgesi.shape[0] < 20 or insan_bolgesi.shape[1] < 20:
                continue
                
            # 2. Aşama: Çıkarılan bölge üzerinde kişisel koruyucu donanım (KKD) analizi
            ekipman_sonuclar = ekipman_modeli.predict(source=insan_bolgesi, verbose=False)
            
            baret_var_mi = False
            yelek_var_mi = False
            
            # Sınıflandırma sonuçlarının değerlendirilmesi
            for es in ekipman_sonuclar:
                bulunan_siniflar = es.boxes.cls
                if 0 in bulunan_siniflar: # Sınıf 0: Baret (Safety-Helmet)
                    baret_var_mi = True
                if 1 in bulunan_siniflar: # Sınıf 1: Yelek (Reflective-Jacket)
                    yelek_var_mi = True
                    
            # Karar mekanizması ve görselleştirme
            if baret_var_mi and yelek_var_mi:
                # Kurallara uygun durum
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Guvenli", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # İhlal durumu
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                eksikler = []
                if not baret_var_mi: eksikler.append("Baret Yok")
                if not yelek_var_mi: eksikler.append("Yelek Yok")
                
                uyari_metni = "IHLAL: " + ", ".join(eksikler)
                cv2.putText(frame, uyari_metni, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Aynı kişi için çoklu fotoğraf kaydını önlemek adına 2 saniyelik bekleme süresi
                if su_an - son_fotograf_zamani > 2:
                    zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dosya_adi = f"ihlaller/ihlal_{zaman}.jpg"
                    cv2.imwrite(dosya_adi, frame)
                    print(f"DIKKAT: İhlal tespit edildi. Kayıt: {dosya_adi}")
                    son_fotograf_zamani = su_an

    # İşlenmiş görüntünün ekrana yansıtılması
    cv2.imshow("Santiye Guvenlik Sistemi", frame)
    
    # 'q' tuşu ile manuel çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Sistem kullanıcı tarafından kapatılıyor...")
        break

# Kaynakların serbest bırakılması
kamera.release()
cv2.destroyAllWindows()
