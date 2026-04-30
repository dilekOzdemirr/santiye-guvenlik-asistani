# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:24:22 2026

@author: Admin
"""

from ultralytics import YOLO

def main():
    # 1. Önceden eğitilmiş en hafif ve hızlı modeli (nano) indir
    print("Model yükleniyor...")
    model = YOLO('yolov8n.pt') 

    # 2. Modeli kendi veri setimizle eğitmeye başla
    print("Eğitim başlıyor! Arkanıza yaslanın...")
    results = model.train(
        data='datasets/data.yaml', # Az önce koyduğun dosyanın yolu
        epochs=50,                 # Modelin veriler üzerinden kaç tur geçeceği
        imgsz=640,                 # Resimlerin eğitim boyutu
        batch=16,                  # Belleğe tek seferde gidecek resim sayısı
        name='santiye_modeli',     # Kaydedilecek klasörün adı
        device=0               # Eğitimi işlemci (CPU) ile yapıyoruz
    )
    
    print("Eğitim başarıyla tamamlandı!")

if __name__ == '__main__':
    main()