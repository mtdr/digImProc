import cv2
import numpy as np
import sys


def main(image):
    if image is None:
        print("File name in current directory ")
        way = input()
        image = cv2.imread(way)

    output = image.copy()
    print("Distance between circles")
    arg1 = float(input())
    print("Min. radius")
    arg2 = float(input())
    print("Max. radius")
    arg3 = float(input())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, arg1, arg2, arg3)

    # ensure at least some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imwrite('result.bmp', output)
        cv2.waitKey(0)


if len(sys.argv) == 2:
    img = sys.argv[1]
else:
    img = cv2.imread("circles.jpg")

main(img)
