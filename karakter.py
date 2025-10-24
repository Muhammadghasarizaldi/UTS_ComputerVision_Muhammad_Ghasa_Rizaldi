import cv2
import numpy as np
import os

# Buat folder output 
os.makedirs("output", exist_ok=True)


# 1. Membuat karakter Teddie

# Kanvas putih
canvas = np.full((500, 400, 3), 255, dtype=np.uint8)

# Kepala (biru)
cv2.circle(canvas, (200, 180), 100, (255, 0, 0), -1)

# Wajah (putih)
cv2.circle(canvas, (200, 180), 80, (255, 255, 255), -1)

# Telinga
cv2.circle(canvas, (120, 90), 30, (255, 0, 0), -1)
cv2.circle(canvas, (280, 90), 30, (255, 0, 0), -1)

# Mata
cv2.circle(canvas, (170, 170), 10, (0, 0, 0), -1)
cv2.circle(canvas, (230, 170), 10, (0, 0, 0), -1)

# Mulut
cv2.rectangle(canvas, (170, 210), (230, 220), (0, 0, 0), -1)

# Tubuh (merah)
cv2.rectangle(canvas, (150, 280), (250, 420), (0, 0, 255), -1)

# Tombol baju
for y in [310, 350, 390]:
    cv2.circle(canvas, (200, y), 8, (255, 255, 255), -1)



cv2.imwrite("output/karakter.png", canvas)


# 2. Transformasi

# Translasi (geser)
M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(canvas, M_trans, (400, 500))
cv2.imwrite("output/translate.png", translated)

# Rotasi
rows, cols = canvas.shape[:2]
M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 25, 1)
rotated = cv2.warpAffine(canvas, M_rot, (cols, rows))
cv2.imwrite("output/rotate.png", rotated)

# Resize (perkecil)
resized = cv2.resize(canvas, None, fx=0.6, fy=0.6)
cv2.imwrite("output/resize.png", resized)

# Crop (bagian wajah)
cropped = canvas[100:300, 100:300]
cv2.imwrite("output/crop.png", cropped)


# 3. Operasi Bitwise

# Buat background (bisa juga diganti dengan gambar lain)
bg = np.full((500, 400, 3), (200, 255, 200), dtype=np.uint8)

# Mask & inverse mask
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)

# Ambil bagian karakter & background
fg = cv2.bitwise_and(canvas, canvas, mask=mask)
bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)

# Gabungkan
bitwise_result = cv2.add(bg_part, fg)
cv2.imwrite("output/bitwise.png", bitwise_result)


# 4. Gambar Akhir (Final)

# Tambahkan teks
final = bitwise_result.copy()
cv2.putText(final, "Persona 4 - Teddie", (40, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,80), 2)
cv2.imwrite("output/final.png", final)


# 5. Tampilkan hasil

cv2.imshow("Karakter Asli", canvas)
cv2.imshow("Translasi", translated)
cv2.imshow("Rotasi", rotated)
cv2.imshow("Resize", resized)
cv2.imshow("Crop", cropped)
cv2.imshow("Bitwise", bitwise_result)
cv2.imshow("Final", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
