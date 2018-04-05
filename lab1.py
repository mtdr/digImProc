from math import *
import cv2
import numpy as np


def stalker(image, res):
    (h, w) = resized.shape[:2]
    center = (w / 2, h / 2)


def findCoord(image, angle):
    (h, w) = image.shape[:2]
    center = [w / 2, h / 2]
    pointC = [center[0] + w / 2, center[1]]
    lineAB = lineAC = L(pointC[0], pointC[1], center[0], center[1])
    a_lambda = (180 - angle) / 2
    lineBC = lineAB * sin(angle) / sin(a_lambda)
    lineBE = lineBC * sin(a_lambda) / sin(90)
    lineEC = lineBC * sin(90 - a_lambda) / sin(90)
    pointB = [(pointC[0] - lineEC), (pointC[1] - lineBE)]
    return [pointB, pointC]


def L(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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

angle = 70
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(resized, M, (w, h))

rect = np.zeros(rotated.shape, rotated.dtype)
res = findCoord(resized, angle)
print(res)
cv2.imshow("Rotated image", rotated)
cv2.waitKey(0)

# cropped = resized[30:130, 150:300]
# cv2.imshow("Cropped image", cropped)
# cv2.waitKey(0)
