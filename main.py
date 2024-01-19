import cv2
import numpy as np

# Kamera aç
cap = cv2.VideoCapture(0)  # 0, bilgisayarınıza bağlı bir kamerayı ifade eder. Birden fazla kamera varsa değiştirilebilir.

while True:
    # Kameradan bir frame al
    ret, frame = cap.read()

    # Eğer frame alınamazsa döngüyü sonlandır
    if not ret:
        print("Hata: Video akışı alınamıyor.")
        break

    # Görüntüyü gri tonlamaya çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Görüntüyü blurla
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Kenarları tespit et
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Şerit alanlarını maskele
    mask = np.zeros_like(edges)
    mask_roi = np.array([[(100, frame.shape[0]), (400, 300), (600, 300), (900, frame.shape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, mask_roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Dönüşümü uygula
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

    # Şeritleri görselleştir
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Orijinal ve işlenmiş görüntüyü ekranda göster
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)

    # Eğer 'q' tuşuna basılırsa döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()

# Pencereleri kapat
cv2.destroyAllWindows()
