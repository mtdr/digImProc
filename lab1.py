from math import *
import cv2
import numpy as np

x_shift = 0
y_shift = 0


def getCenter(matrix):
    (h, w) = matrix.shape[:2]
    return w / 2, h / 2


def my_round(var):
    return [int(round(var[0])), int(round(var[1]))]


def findCoord(point, alpha, image):
    point_c = point
    center = getCenter(image)
    lineAC = L(point_c[0], point_c[1], center[0], center[1])
    gamma = (180 - alpha) / 2
    lineBC = sqrt(2 * pow(lineAC, 2) * (1 - cos(radians(alpha))))
    # lineBC = lineAC * sin(angle_f) / sin(a_lambda)
    # lineBE = lineBC * sin(a_lambda) / sin(90)
    # lineEC = lineBC * sin(90 - a_lambda) / sin(90)
    # pointB = [(point_c[0] - lineEC), (point_c[1] - lineBE)]
    var1 = (lineBC * cos(radians(alpha)))
    var2 = (lineBC * sin(radians(alpha)))
    if (point[0] >= center[0]) & (point[1] <= center[1]):
        pointB = [point[0] - var1, point[1] - var2]

    if (point[0] > center[0]) & (point[1] > center[1]):
        pointB = [point[0] + var1, point[1] - var2]

    if (point[0] < center[0]) & (point[1] > center[1]):
        pointB = [point[0] + var1, point[1] + var2]

    if (point[0] < center[0]) & (point[1] < center[1]):
        pointB = [point[0] - var1, point[1] + var2]

    return my_round(pointB)


def addToRes(point, res):
    x_s, y_s = res.shape[0] - res.shape[0] / 4  # start point; analog 0,0
    res[x_s + x_shift][y_s + y_shift] = point


def add_points(image, angle):
    size = image.shape[1], image.shape[0]
    # resSize = size[0]
    # if size[1] > resSize:
    # resSize = size[1]

    # res = np.zeros([resSize * 2, resSize * 2], int)  # if angle = 90 w <=> h
    res = []
    for i in range(0, size[0] - 1, 1):
        res.append(findCoord([i, 0], angle, image))

    # for i in range(0, size[1] - 1, 1):
    #     res.append(findCoord([size[0], i], angle, image))

    # for i in reversed(range(1, size[0], 1)):
    #     res.append(findCoord([i, size[1]], angle, image))
    #
    # for i in range(0, size[1] - 1, 1):
    #     res.append(findCoord([size[0], i], angle, image))
    print(image.shape)
    print(res)
    return res


def getResult(p0, an_alpha, image):
    print(p0)
    print(findCoord(p0, an_alpha, image))


def L(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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
    cv2.imshow("Resize image", resized)
    cv2.waitKey(0)

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
    # getResult([0, 0], angle, rotated)
    add_points(rotated, angle)
    cv2.imshow("Rotated image", rotated)
    cv2.waitKey(0)

    # cropped = rotated[30:130, 150:300]
    # cropped = rotated[0:50, 375:499]
    cropped = rotated[0:398, 125:500]
    cv2.imshow("Cropped image", cropped)
    cv2.waitKey(0)


main()
