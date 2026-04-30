# -*- coding: utf-8 -*-
"""
Created on Fri May  1 02:13:15 2026

@author: Admin
"""

from ultralytics import YOLO

def main():
    print(" model yükleniyor...")
    # 40 turun sonunda kaydedilen en iyi ağırlıkları çağırıyoruz
    model = YOLO('runs/detect/santiye_modeli-4/weights/best.pt')
    
    # Test edilecek resmin yolu. 
    # Not: Proje klasörüne test_resmi.jpg adında bir şantiye fotoğrafı koymalısın!
    resim_yolu = 'test_resmi.jpg' 
    
    print("Resim analiz ediliyor...")
    # show=True ekranda anında gösterir, save=True ise runs/detect/predict klasörüne kaydeder
    results = model.predict(source=resim_yolu, show=True, save=True)
    
    print("Test tamamlandı!")

if __name__ == '__main__':
    main()