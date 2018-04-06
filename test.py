import numpy as np

test = np.zeros([3, 5], int)
for j in range(0, test.shape[1], 1):
    for i in range(0, test.shape[0], 1):
        test[i][j] = -1
print(test)

# rotatedM = np.ones([rotated.shape[0], rotated.shape[1]], int)

# test = np.zeros([3, 5], int)
# for j in range(0, test.shape[1], 1):
#     for i in range(0, test.shape[0], 1):
#         test[i][j] = -1
# for i in range(0, test.shape[0], 1):
#     test[i][2] = test[i][3] = 1
# test[1][4] = 1


# def main():
#     image = cv2.imread("big.jpg")
#     angle = 350
#
#     # Нам надо сохранить соотношение сторон
#     # чтобы изображение не исказилось при уменьшении
#     # для этого считаем коэф. уменьшения стороны
#     final_wide = 500
#     r = float(final_wide) / image.shape[1]
#     dim = (final_wide, int(image.shape[0] * r))
#
#     # уменьшаем изображение до подготовленных размеров
#     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     # cv2.imshow("Resize image", resized)
#     # cv2.waitKey(0)
#
#     # получим размеры изображения для поворота
#     # и вычислим центр изображения
#     (h, w) = resized.shape[:2]
#     center = (w / 2, h / 2)
#
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(resized, M, (w, h))
#
#     p_right_center = [center[0] + w / 2, center[1]]
#     # getResult(p_right_center, angle)
#     # getResult([rotated.shape[1], rotated.shape[0]], angle, rotated)
#     # getResult([0, h], angle, rotated)
#
#     new_borders = add_points(rotated, angle)
#     filtered = filter_border(new_borders, w, h)
#     onematrix = getNewMatrix(rotated, filtered)
#     filled = fill_matr(onematrix)
#     result = intel_alg(filled)
#     result1 = c_alg(filled)
#
#
#     cv2.imshow("Rotated image", rotated)
#     cv2.waitKey(0)
#
#     cropped = rotated[result[1][0]:result[2][0], result[1][1]:result[2][1]]
#     cv2.imshow("Cropped image", cropped)
#     cv2.waitKey(0)
#
