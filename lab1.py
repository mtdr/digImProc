from math import *

import cv2
import numpy as np
import sys


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
    # print(image.shape)
    # print(len(res))
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
        if x >= w:
            temp[0] = w - 1
        if y < 0:
            temp[1] = 0
        if y >= h:
            temp[1] = h - 1
        res.append(temp)
    return res


def getNewMatrix(image, borders):
    res = np.zeros([image.shape[0], image.shape[1]], int)
    for j in range(0, image.shape[1], 1):
        for i in range(0, image.shape[0], 1):
            res[i][j] = -1
    cr = 0
    size = image.shape[1], image.shape[0]
    for j in range(0, size[0], 1):
        for i in range(0, size[1], 1):
            if [j, i] in borders:
                res[i][j] = 1
                cr += 1
    print(cr)
    return res


def findRect(matrix):
    n, m = matrix.shape[:2]  # check order
    a = matrix
    ans = 0
    d = [0] * m
    d1 = d2 = d
    st = []
    for i in range(0, n, 1):
        for j in range(0, m, 1):
            if a[i][j] == 1:
                d[j] = i
        st.clear()
        for j in range(0, m, 1):
            while len(st) > 0:
                if d[st[-1]] <= d[j]:
                    st.pop()
            if len(st) == 0:
                d1[j] = -1
            else:
                d1[j] = st[-1]
            st.append(j)
        st.clear()
        for j in reversed(range(0, m, 1)):
            while len(st) > 0:
                if d[st[-1]] <= d[j]:
                    st.pop()
            if len(st) == 0:
                d2[j] = m
            else:
                d2[j] = st[-1]
            st.append(j)
        for j in range(0, m, 1):
            ans = max(ans, (i - d[j]) * (d2[j] - d1[j] - 1))
            if ans > 0:
                print(ans)
    return ans


def cadane(matrix):
    M = 0
    P1 = P2 = [0, 0]
    n, m = matrix.shape[:2]
    a = matrix

    for g in range(0, n, 1):
        p = [0] * m
        for i in range(g, n, 1):
            t = 0
            h = 0
            for j in range(0, m, 1):
                if a[i][j] == 1:
                    p[j] = p[j] + a[i][j]
                    t = t + p[j]
                    if t > M:
                        M = t
                        P1 = [g, h]
                        P2 = [i, j]
                    if t <= 0:
                        t = 0
                        h = j + 1 # intel error
                else:
                    t = 0
                    h = j + 1
    return M, P1, P2


def mssl(x):
    max_ending_here = max_so_far = 0
    for a in x:
        max_ending_here = max(0, max_ending_here + a)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def kad_c(arr, start, finish, n):
    sumK = 0
    max_sum = -sys.maxsize - 1
    finish = -1
    local_start = 0
    for i in range(0, n, 1):
        if arr[i] == 1:
            sumK += arr[i]
            if sumK < 0:
                sumK = 0
                local_start = i + 1
            else:
                if sumK > max_sum:
                    max_sum = sumK
                    start = local_start
                    finish = i

    if finish != -1:
        return max_sum, start, finish

    # max_sum = arr[0]
    # start = finish = 0
    # for i in range(1, n, 1):
    #     if arr[i] > max_sum:
    #         max_sum = arr[i]
    #         start = finish = i
    #
    # return max_sum, start, finish


def find_max(matrix):
    ROW, COL = matrix.shape[:2]
    max_sum = -sys.maxsize - 1
    finalLeft, finalRight, finalTop, finalBottom, left, right, it1, sum1, start, finish = [0] * 10
    for left in range(0, COL, 1):
        temp = [0] * ROW
        for right in range(left, COL, 1):
            for it1 in range(0, ROW, 1):
                temp[it1] = matrix[it1][right]
            sum1, start, finish = kad_c(temp, start, finish, ROW)

            if sum1 > max_sum:
                max_sum = sum1
                finalLeft = left
                finalRight = right
                finalTop = start
                finalBottom = finish
    print("Top Left y0 = ", finalTop, " x0 = ", finalLeft)
    print("Bottom Right y1 = ", finalBottom, " x1 = ", finalRight)
    print("Max val is ", max_sum)


def fill_matr(matrix):
    res = matrix
    for i in range(0, res.shape[0], 1):
        first, last = 0, 0
        for j in range(0, res.shape[1], 1):
            if res[i][j] == 1:
                first = j
                break
        for j in range(0, res.shape[1], 1):
            if res[i][j] == 1:
                last = j
        if first != last:
            for k in range(first, last, 1):
                res[i][k] = 1
    return res


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
    onematrix = getNewMatrix(rotated, filtered)
    filled = fill_matr(onematrix)
    coords = work(filled)
    # rotatedM = np.ones([rotated.shape[0], rotated.shape[1]], int)

    # test = np.zeros([3, 5], int)
    # for j in range(0, test.shape[1], 1):
    #     for i in range(0, test.shape[0], 1):
    #         test[i][j] = -1
    # for i in range(0, test.shape[0], 1):
    #     test[i][2] = test[i][3] = 1
    # test[1][4] = 1

    cv2.imshow("Rotated image", rotated)
    cv2.waitKey(0)

    # cropped = rotated[30:130, 150:300]
    # cropped = rotated[0:50, 375:499]
    cropped = rotated[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1]]
    cv2.imshow("Cropped image", cropped)
    cv2.waitKey(0)


def work(matr):
    c = cadane(matr)
    print(c)
    print(matr.shape)
    print(find_max(matr))
    return c[1], c[2]


main()
