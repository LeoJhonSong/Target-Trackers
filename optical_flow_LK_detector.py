'''
基于Lucas-Kanade算法的稀疏光流法目标跟踪器

USAGE: optical_flow_GF_detector.py [<video_source>]
Keys:
    ESC    - exit
'''

import sys
import time
import numpy as np
import cv2
from utils import Recorder


if __name__ == '__main__':
    try:
        source = sys.argv[1]
    except IndexError:
        source = 0

    cap = cv2.VideoCapture(source)
    ret, old_frame = cap.read()

    # 创建recorder
    recorder = Recorder(cap, './log/optical_flow_LK')

    # ShiTomasi corner detection的参数
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # 光流法参数
    # maxLevel 未使用的图像金字塔层数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机生成的颜色
    color = np.random.randint(0, 255, (100, 3))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片
    start = time.time()

    while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算光流以获取点的新位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 选择good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # 绘制跟踪框
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        recorder.write(img, '')
        cv2.imshow('frame', img)
        k = cv2.waitKey(30)  # & 0xff
        if time.time() - start > 2:
            start = time.time()
            mask = np.zeros_like(old_frame)
        # time.sleep(0.1)
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    recorder.release()
    cap.release()
    cv2.destroyAllWindows()
