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

# === 动作分割函数 ===
def detect_actions(frame_changes, static_threshold=7, static_min_frames=20, dynamic_min_frames=2):
    actions = []
    in_action = False  # 是否在动作中
    max_change_frame = None  # 记录 frame_changes 最大的帧
    static_count = 0  # 统计静止帧

    for i in range(len(frame_changes) - 1):
        if frame_changes[i] > static_threshold:  # 动作区域
            if not in_action:
                start_frame = i
                in_action = True
            if max_change_frame is None or (frame_changes[i] > frame_changes[max_change_frame] and static_count < static_min_frames):
                max_change_frame = i  # 只有在非静止状态下更新最大变化点
            static_count = 0  # 重置静止计数
        else:  # 静止区域
            static_count += 1
            if static_count >= static_min_frames and in_action:
                actions.append((start_frame, max_change_frame))  # 结束动作
                in_action = False
                static_count = 0
                max_change_frame = None

    # 确保最后一个动作不会丢失
    if in_action:
        actions.append((start_frame, len(frame_changes) - 1))

    return actions

# === 读取数据 ===
skeleton_sequences = np.load("joint_positions19.npy")  # 示例数据：100帧，17个关键点，3D坐标
timestampes = np.load("17.npy")  # 示例数据：100帧的时间戳
print(timestampes.shape, skeleton_sequences.shape)
# 获取变化
frame_changes = calculate_frame_changes(skeleton_sequences)
frame_indices = np.arange(len(frame_changes))  # 帧索引

# 进行动作分割（基于静止与动态变化）
actions = detect_actions(frame_changes, static_threshold=6.5, static_min_frames=30)

# === 绘制曲线 ===
plt.figure(figsize=(12, 5))
plt.plot(frame_indices, frame_changes, label="Frame Changes", color="blue", linewidth=1.5)

# 标注动作区域
for start, end in actions:
    plt.axvspan(start, end, color="green", alpha=0.3)  # 绿色阴影代表动作区域

plt.xlabel("Frame Index")
plt.ylabel("Change Magnitude")
plt.title("Frame Change Curve with Action Segments")
plt.legend()
plt.grid(True)
plt.show()

# === 输出动作片段信息 ===
print("Detected actions:", actions)
for start, end in actions:
    print(f"动作: {start} - {end}（持续 {end - start} 帧）")