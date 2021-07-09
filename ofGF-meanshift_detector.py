"""
基于Gunnar_Farneback算法框选目标, MeanShift算法跟踪的目标跟踪器

USEAGE: python ofGF-meanshift_detector.py [<video_source>]
Keys:
    ESC - exit
"""

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


def ofGF(old_gray, new_img):
    min_distance = 5
    thr = 0.05  # 目标阈值框padding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 5, 25, 3, 7, 1.5, 0)
    # 归一化
    h, w = flow.shape[:2]
    norms = np.linalg.norm(flow, axis=2)
    norms = norms / max(30, norms.max()) * 255  # 缩放到255尺度
    norms = norms.astype(np.uint8)
    norms[np.logical_and(-1 * min_distance < norms, norms < min_distance)] = 0
    norms = np.abs(norms)
    # 滤波
    norms = cv2.GaussianBlur(norms, (7, 7), 0)
    ret, thresh = cv2.threshold(norms, 40, 255, cv2.THRESH_BINARY)
    bin = cv2.erode(thresh, kernel, iterations=2)
    bin = cv2.dilate(bin, kernel2, iterations=3)
    # 选取最大且大于一定面积的轮廓
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bin = np.full((bin.shape[0], bin.shape[1], 3), 0, dtype=np.uint8)
    if len(contours):
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
        if areas[areas_s[0][0]] > h * w / 300:
            cv2.drawContours(img_bin, contours, areas_s[0][0], [255, 255, 0], -1)
    # 计算目标轮廓重心 (已归一化)
    bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(bin)
    info = ''
    locked = 0
    track_window = None
    if moments['m00'] != 0:  # 如果有目标
        cx = moments['m10'] / moments['m00'] / w
        cy = moments['m01'] / moments['m00'] / h
        info = "x: %.6f, y: %.6f" % (cx, cy)
        if cv2.countNonZero(bin) == cv2.countNonZero(bin[int(h * thr):-int(h * thr), int(w * thr):-int(w * thr)]):
            locked = 1
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            track_window = cv2.boundingRect(contours[0])
    # 判断目标是否进入阈值框, 是的话给出给出框选框
    img_bin = cv2.bitwise_not(img_bin, dst=None)  # 反色, 转为红色目标白色背景
    # 将标红的目标融合到原图像
    b_bin, g_bin, r_bin = cv2.split(img_bin)
    b, g, r = cv2.split(new_img)
    new_img = cv2.merge([b & b_bin, g & g_bin, r & r_bin])
    return new_gray, new_img, info, locked, track_window


def meanshift(roi_hist, img, track_window):
    # 设置终止条件
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # apply meanshift to get the new location
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  # 计算反向投影图
    _, track_window = cv2.meanShift(dst, track_window, term_crit)
    topLeft_x, topLeft_y, width, height = track_window
    # 框选目标
    cv2.rectangle(img, (topLeft_x, topLeft_y), (topLeft_x + width, topLeft_y + height), 255, 2)
    # 标出目标中心点
    cv2.rectangle(img, (int(topLeft_x + width / 2), int(topLeft_y + height / 2)),
                    (int(topLeft_x + width / 2), int(topLeft_y + height / 2)), (0, 0, 255), 5)
    cx = (topLeft_x + width / 2) / img.shape[1]
    cy = (topLeft_y + height / 2) / img.shape[0]
    info = '坐标: ' + str(cx) + ', ' + str(cy)

    return track_window, img, info


if __name__ == '__main__':
    import sys
    from utils import Recorder
    try:
        source = sys.argv[1]
    except IndexError:
        source = 0

    cap = cv2.VideoCapture(source)
    ret, old = cap.read()
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    locked = 0
    track_window = []
    roi_hist = None
    # 创建recorder
    recorder = Recorder(cap, './log/ofGF-meanshift')
    while(cap.isOpened()):
        ret, img = cap.read()
        if cv2.waitKey(1) == 27 or not ret:
            break
        if not locked:
            old_gray, img, info, locked, track_window = ofGF(old_gray, img)
        elif locked == 1:
            locked = 2
            info = 'target locked'
            topLeft_x, topLeft_y, width, height = track_window
            roi = img[topLeft_y: topLeft_y + height, topLeft_x: topLeft_x + width]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        else:
            track_window, img, info = meanshift(roi_hist, img, track_window)
        recorder.write(img, info)
        cv2.imshow('img', img)
    recorder.release()
    cap.release()
    cv2.destroyAllWindows()
