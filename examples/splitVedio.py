import cv2
import numpy as np
import subprocess

# 读取 MKV 文件
video_path = '10.mkv'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("错误：无法打开 MKV 文件！")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义动作检测参数
motion_threshold = 8000  # 帧间差异阈值
min_static_frames = int(fps * 1)  # 至少2秒的静止时间

# 初始化变量
prev_frame = None
static_frames = 0
action_start = 0
actions = []

# 遍历视频帧
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("视频读取完成或帧读取失败！")
        break

    # 转换为灰度图并模糊处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 如果是第一帧，初始化 prev_frame
    if prev_frame is None:
        prev_frame = gray
        continue

    # 计算帧间差异
    frame_diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    diff_sum = np.sum(thresh)
    # print(f"Frame {frame_idx}: Diff Sum = {diff_sum}")

    # 判断是否静止
    if diff_sum < motion_threshold:
        static_frames += 1
    else:
        # 如果静止时间超过阈值，记录动作结束时间
        if static_frames >= min_static_frames:
            if action_start > 0:  # 确保不是视频开头
                actions.append((action_start, frame_idx - static_frames))
            action_start = frame_idx  # 新动作的开始时间
        static_frames = 0

    prev_frame = gray
    frame_idx += 1

# 添加最后一个动作
if action_start > 0:
    actions.append((action_start, frame_idx))

cap.release()

# 输出动作时间段
if not actions:
    print("未检测到任何动作片段！")
else:
    for i, (start_frame, end_frame) in enumerate(actions):
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        print(f"Action {i+1}: {start_sec:.2f}s - {end_sec:.2f}s")
