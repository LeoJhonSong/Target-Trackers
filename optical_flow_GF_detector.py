#!/usr/bin/env python

'''
基于Gunnar_Farneback算法的稠密光流法

USAGE: optical_flow_GF_detector.py [<video_source>]
Keys:
    1 - toggle HSV flow visualization
    ESC    - exit
'''

import numpy as np
import cv2


def draw_flow(img, gray, flow, step=16):
    h, w = gray.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    scale = 0.8
    cap = cv2.VideoCapture(fn)
    ret, prev = cap.read()
    h, w = prev.shape[0:2]
    prev = cv2.resize(prev, (int(w * scale), int(h * scale)))
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    min_distance = 5
    step = 16
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while(cap.isOpened()):
        ret, img = cap.read()
        h, w = img.shape[0:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.8, 5, 30, 3, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 5, 25, 3, 7, 1.5, 0)
        prevgray = gray
        # get norms
        h, w = flow.shape[:2]
        norms = np.linalg.norm(flow, axis=2)
        norms = norms / max(30, norms.max()) * 255  # 缩放到255尺度
        norms = norms.astype(np.uint8)
        norms[np.logical_and(-1 * min_distance < norms, norms < min_distance)] = 0
        norms = np.abs(norms)
        # some operate
        norms = cv2.GaussianBlur(norms, (7, 7), 0)
        # cv2.imshow('flow norm', norms)
        ret, thresh = cv2.threshold(norms, 40, 255, cv2.THRESH_BINARY)
        bin = cv2.erode(thresh, kernel, iterations=2)
        bin = cv2.dilate(bin, kernel2, iterations=3)
        # cv2.imshow('bin', bin)
        # img_bin = cv2.cvtColor(cv2.bitwise_not(bin, dst=None), cv2.COLOR_GRAY2BGR)  # 反色, 以白色为背景色
        img_bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 选取最大且大于一定面积的轮廓
        if len(contours):
            areas = np.array([cv2.contourArea(contour) for contour in contours])
            areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
            img_bin = np.full((bin.shape[0], bin.shape[1], 3), 0, dtype=np.uint8)
            if areas[areas_s[0][0]] > h * w / 300:
                cv2.drawContours(img_bin, contours, areas_s[0][0], [255, 255, 0], -1)
        # cv2.imshow('img_bin', img_bin)
        # 计算目标轮廓重心 (已归一化)
        moments = cv2.moments(cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY))
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00'] / w
            cy = moments['m01'] / moments['m00'] / h
            print("x: %.6f, y: %.6f" % (cx, cy))
        img_bin = cv2.bitwise_not(img_bin, dst=None)
        # 将标红的目标融合到原图像
        b_bin, g_bin, r_bin = cv2.split(img_bin)
        b, g, r = cv2.split(img)
        img = cv2.merge([b & b_bin, g & g_bin, r & r_bin])
        # cv2.imshow('flow', draw_flow(img, gray, flow))
        cv2.imshow('flow', img)
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        ch = cv2.waitKey(1)
        if ch == 27 or not ret:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
    cv2.destroyAllWindows()
    cap.release()
