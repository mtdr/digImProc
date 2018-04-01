import cv2
import numpy as np

image = cv2.imread("tram.jpg")
# Нам надо сохранить соотношение сторон
# чтобы изображение не исказилось при уменьшении
# для этого считаем коэф. уменьшения стороны
final_wide = 500
r = float(final_wide) / image.shape[1]
dim = (final_wide, int(image.shape[0] * r))

# уменьшаем изображение до подготовленных размеров
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow("Resize image", resized)
# cv2.waitKey(0)

# получим размеры изображения для поворота
# и вычислим центр изображения
(h, w) = resized.shape[:2]
center = (w / 2, h / 2)

M = cv2.getRotationMatrix2D(center, 70, 1.0)
rotated = cv2.warpAffine(resized, M, (w, h))

rect = np.zeros(rotated.shape, rotated.dtype)

cv2.imshow("Rotated image", rotated)
cv2.waitKey(0)

# cropped = resized[30:130, 150:300]
# cv2.imshow("Cropped image", cropped)
# cv2.waitKey(0)


def stalker(image, res):
    (h, w) = resized.shape[:2]
    center = (w / 2, h / 2)


