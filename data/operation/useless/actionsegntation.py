import numpy as np
import matplotlib.pyplot as plt

# 计算帧间骨骼点变化
def calculate_frame_changes(skeleton_sequences):
    changes = []
    for i in range(1, len(skeleton_sequences)):
        # 计算每一帧与前一帧的欧氏距离
        diff = np.linalg.norm(skeleton_sequences[i] - skeleton_sequences[i-1], axis=1)
        changes.append(np.mean(diff))  # 取所有关键点的平均变化
    return np.array(changes)

# 动作分割
def detect_actions(frame_changes, min_gap=30, min_duration=30):
    mean_change = np.mean(frame_changes)
    std_change = np.std(frame_changes)
    threshold = mean_change + 0.3 * std_change  # 自适应阈值
    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.plot(frame_changes, label="Frame Changes", color="blue", linewidth=1.5)
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")

    # 标注动作区域
    for i, change in enumerate(frame_changes):
        if change > threshold:
            plt.scatter(i, change, color="red", s=10)  # 画出超过阈值的点

    plt.xlabel("Frame Index")
    plt.ylabel("Change Magnitude")
    plt.title("Frame Change Curve with Adaptive Threshold")
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()
    actions = []
    i = 0
    while i < len(frame_changes):
        if frame_changes[i] > threshold:
            start_frame = i
            while i < len(frame_changes) and frame_changes[i] > threshold:
                i += 1
            end_frame = i
            if (end_frame - start_frame) >= min_duration:  # 确保动作至少持续一定时间
                if not actions or (start_frame - actions[-1][1] > min_gap):
                    actions.append((start_frame, end_frame))
        else:
            i += 1
    return actions, threshold


# 假设skeleton_sequences是骨骼点序列，形状为 (帧数, 关键点数, 坐标维度)
skeleton_sequences = np.load("joint_positions1.npy")  # 示例数据：100帧，17个关键点，3D坐标

timestampes = np.load("16.npy")  # 示例数据：100帧的时间戳
print(skeleton_sequences.shape, timestampes.shape)
# 获取变化
frame_changes = calculate_frame_changes(skeleton_sequences)
print("Max change:", np.max(frame_changes))
print("Min change:", np.min(frame_changes))
# 动作分割
actions, threshold = detect_actions(frame_changes)
print("Detected actions:", actions)
for item in actions:
    start, end = item  # 解包元组
    print(f"起始值: {timestampes[start]}, 结束值: {timestampes[end]}, 持续时间: {timestampes[end] - timestampes[start]}")
