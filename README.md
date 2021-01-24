# Target Tracker

- `meanshift_detector.py`为基于MeanShift算法的手动框选目标跟踪器
- `frame_difference_detector.py`为基于帧差法的目标跟踪器
- `optical_flow_LK_detector.py`为基于Lucas-Kanade算法的稀疏光流法目标跟踪器
- `optical_flow_GF_detector.py`为基于Gunner_Farneback算法的稠密光流法目标跟踪器

## 运行

要运行`optical_flow_GF_detector.py`的话, 执行`python optical_flow_GF_detector.py`以默认摄像头为视频源运行; 执行`python optical_flow_GF_detector.py ./test.mkv`以当前文件夹下`test.mkv`为视频源运行.