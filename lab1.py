from math import *

import cv2
import numpy as np


def getCenter(matrix):
    (h, w) = matrix.shape[:2]
    return w / 2, h / 2


def my_round(var):
    return [int(round(var[0])), int(round(var[1]))]


# Функция поиска новых координат точек, используется матрица поворота и сдвига точки вращения
def findCoord(point, alpha, image):
    center = getCenter(image)
    pointB1 = [(point[0] - center[0]) * cos(radians(alpha)) + (point[1] - center[1]) * sin(radians(alpha)) + center[0],
               -(point[0] - center[0]) * sin(radians(alpha)) + (point[1] - center[1]) * cos(radians(alpha)) + center[1]]
    return my_round(pointB1)


# Ищем новые координаты границ рисунка, получаем 4х угольник
def add_points(image, angle):
    size = image.shape[1], image.shape[0]
    res = []
    for i in range(0, size[0], 1):
        res.append(findCoord([i, 0], angle, image))

    for i in range(0, size[1], 1):
        res.append(findCoord([size[0] - 1, i], angle, image))

    for i in reversed(range(0, size[0], 1)):
        res.append(findCoord([i, size[1] - 1], angle, image))

    for i in reversed(range(0, size[1], 1)):
        res.append(findCoord([0, i], angle, image))

    # for j in range(0, size[1] + 1, 1):
    # for i in range(0, size[0] + 1, 1):
    # res.append(findCoord([i, j], angle, image))
    print(image.shape)
    print(len(res))
    return res


def getResult(p0, an_alpha, image):
    print(p0)
    print(findCoord(p0, an_alpha, image))


def L(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Ограничивает границы по размеру окна, получаем многоугольник
def filter_border(borders, w, h):
    res = []
    for i in range(len(borders)):
        x = borders[i][0]
        y = borders[i][1]
        temp = [x, y]
        if x < 0:
            temp[0] = 0
        if x > w:
            temp[0] = w
        if y < 0:
            temp[1] = 0
        if y > h:
            temp[1] = h
        # if i == 0 | (i > 0 & len(list(set(borders[i-1]) & set(temp))) != 0):
        res.append(temp)
    return res


def getNewMatrix(input, borders):
    res = np.zeros(input.shape)
    print(res.shape)


def main():
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

    p_right_center = [center[0] + w / 2, center[1]]
    # getResult(p_right_center, angle)
    # getResult([rotated.shape[1], rotated.shape[0]], angle, rotated)
    # getResult([0, h], angle, rotated)
    new_borders = add_points(rotated, angle)
    filtered = filter_border(new_borders, w, h)
    getNewMatrix(rotated, filtered)

    print(len(filtered))
    cv2.imshow("Rotated image", rotated)
    cv2.waitKey(0)

    # cropped = rotated[30:130, 150:300]
    # cropped = rotated[0:50, 375:499]
    # cropped = rotated[0:398, 125:500]
    # cv2.imshow("Cropped image", cropped)
    # cv2.waitKey(0)


main()
