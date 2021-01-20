# -*- coding: utf-8 -*-
import numpy as np
import cv2


# 框选目标
def set_target(img):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None, 'released_once': False}
    title = 'MeanShift Detector'
    cv2.namedWindow(title)

    def onMouse(event, x, y, flags, mouse_params):
        mouse_params['current_pos'] = (x, y)

        if mouse_params['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            mouse_params['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if mouse_params['tl'] is None:
                mouse_params['tl'] = mouse_params['current_pos']
            elif mouse_params['released_once']:
                mouse_params['br'] = mouse_params['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)  # 设置鼠标事件
    cv2.imshow(title, img)  # 显示最初的画面

    while mouse_params['br'] is None:
        im_draw = np.copy(img)
        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'], mouse_params['current_pos'], (255, 0, 0), 2)
        cv2.imshow(title, im_draw)
        cv2.waitKey(1)

    top_left = ((min(mouse_params['tl'][0], mouse_params['br'][0]), min(mouse_params['tl'][1], mouse_params['br'][1])))
    bottom_right = ((max(mouse_params['tl'][0], mouse_params['br'][0]), max(mouse_params['tl'][1], mouse_params['br'][1])))
    topLeft_x, height, topLeft_y, width = top_left[0], bottom_right[1] - top_left[1], top_left[1], bottom_right[0] - top_left[0]
    return (topLeft_x, topLeft_y, width, height)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 从默认相机读取
    #  cap = cv2.VideoCapture('./test.mkv')

    ret, frame = cap.read()
    track_window = set_target(frame)
    topLeft_x, topLeft_y, width, height = track_window

    # ROI设置
    roi = frame[topLeft_y: topLeft_y + height, topLeft_x: topLeft_x + width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 设置终止条件
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while(cap.isOpened()):
        ret, frame = cap.read()
        # 视频流结束或者按了ESC则结束
        if ret is False or (cv2.waitKey(1) & 0xff == 27):
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  # 计算反向投影图

        # apply meanshift to get the new location
        _, track_window = cv2.meanShift(dst, track_window, term_crit)

        topLeft_x, topLeft_y, width, height = track_window
        cv2.rectangle(frame, (topLeft_x, topLeft_y), (topLeft_x + width, topLeft_y + height), 255, 2)
        cv2.rectangle(frame, (int(topLeft_x + width / 2), int(topLeft_y + height / 2)), (int(topLeft_x + width / 2), int(topLeft_y + height / 2)), (0, 0, 255), 5)
        cv2.imshow('MeanShift Detector', frame)
        cx = (topLeft_x + width / 2) / frame.shape[1]
        cy = (topLeft_y + height / 2) / frame.shape[0]
        print('坐标: ' + str(cx) + ', ' + str(cy))

    cv2.destroyAllWindows()
    cap.release()
