import cv2
import numpy as np
import re


r = r'\.[a-zA-Z0-9]+'


def img_max(image):
    height, width = image.shape[0], image.shape[1]
    result = 0
    for i in range(height):
        for j in range(width):
            if image[i, j] > result:
                result = image[i, j]
    return result


def img_resize(image, k):
    height, width = image.shape[0], image.shape[1]
    result = img_max(image)
    for i in range(height):
        for j in range(width):
            if image[i, j] < result*k:
                image[i, j] = 0
            else:
                image[i, j] = 255
    return image


def edge_detection(image_name):
    image_path = '/home/hero/Documents/DIP-Homework/Homework4/Requirement/{}'.format(image_name)
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[0], image.shape[1]
    global r
    image_name = re.sub(r, '', image_name)
    # Sobel model
    sobel_45_kernel = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]])
    sobel_i45_kernel = np.array([[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
    sobel45 = cv2.filter2D(image, -1, sobel_45_kernel)
    sobeli45 = cv2.filter2D(image, -1, sobel_i45_kernel)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = img_resize(sobely, 0.33)
    sobely = img_resize(sobelx, 0.33)
    sobel45 = img_resize(sobel45, 0.33)
    sobeli45 = img_resize(sobeli45, 0.33)

    sobel = sobeli45 + sobel45 + sobelx + sobely

    canny_edge = cv2.Canny(image, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    canny_edge_dilate = cv2.dilate(canny_edge, kernel)

    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_sobel.jpg'.format(
        image_name), sobel)
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_canny.jpg'.format(
        image_name), canny_edge)
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_canny_dilate.jpg'.format(
        image_name), canny_edge_dilate)


def hough_transform(image_name, method, func):
    image_path = '/home/hero/Documents/DIP-Homework/Homework4/Requirement/{}'.format(image_name)
    image = cv2.imread(image_path)
    global r
    image_name = re.sub(r, '', image_name)
    if method == 'canny':
        canny_path = '/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_canny.jpg'.format(image_name)
        canny_edges = cv2.imread(canny_path)
        gray = cv2.cvtColor(canny_edges, cv2.COLOR_BGR2GRAY)
    elif method == 'canny_dilate':
        canny_dilate_path = '/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_canny_dilate.jpg'.format(
        image_name)
        canny_dilate_edges = cv2.imread(canny_dilate_path)
        gray = cv2.cvtColor(canny_dilate_edges, cv2.COLOR_BGR2GRAY)
    elif method == 'sobel':
        sobel_path = '/home/hero/Documents/DIP-Homework/Homework4/Content/task1/{}_canny_dilate.jpg'.format(
            image_name)
        sobel_edges = cv2.imread(sobel_path)
        gray = cv2.cvtColor(sobel_edges, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if func == 'linesP':
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=2)
        lines1 = lines[:, 0, :]
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif func == 'lines':
        lines = cv2.HoughLines(gray, 200, np.pi / 180, 120)
        lines1 = lines[:, 0, :]
        for rho, theta in lines1[:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework4/Content/task2/{}/{}_hough_{}.jpg'.format(
        func, image_name, method), image)


if __name__ == '__main__':
    image_names = ['test1.tif', 'test2.png', 'test3.jpg', 'test4.bmp', 'test5.png', 'test6.jpg']
    # for image_name in image_names:
    #     edge_detection(image_name)
    hough_transform(image_names[2], 'canny', 'linesP')
