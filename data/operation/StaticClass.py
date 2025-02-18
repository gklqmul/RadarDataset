import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

class ActionSegmenter:
    def __init__(self, max_frame_diff=100, frame_Count=500, min_segment_length=5, long_stationary_threshold=60, smooth_sigma=1.0):
        """
        init ActionSegmenter
        :param threshold_T: threshold for motion energy
        :param min_segment_length: minimum length of action segment
        :param long_stationary_threshold: threshold for long stationary segment
        :param smooth_sigma: parameter for gaussian smoothing
        """
        self.threshold_T = None
        self.min_segment_length = min_segment_length
        self.long_stationary_threshold = long_stationary_threshold
        self.smooth_sigma = smooth_sigma
        self.frame_Count = frame_Count
        self.max_frame_diff = max_frame_diff

    def segment_actions(self, skeleton_frames):
        """
        split the action sequence
        :param skeleton_frames: 骨骼点坐标序列，形状为 (N, M, 3)，N为帧数，M为关节点数，3为(x, y, z)坐标
        :return: 分割后的动作段列表，每个动作段是一个帧索引列表
        """
        # 数据预处理：平滑处理
        skeleton_frames = self._smooth_frames(skeleton_frames)

        # 计算每一帧的整体运动能量
        motion_energy = self._calculate_motion_energy(skeleton_frames)

        # 动态计算阈值 threshold_T
        self.threshold_T = self._calculate_threshold_T(motion_energy)
        
        # 检测静止帧
        is_stationary = motion_energy < self.threshold_T

        # 分割动作序列
        action_segments = self._split_actions(is_stationary)

        # 检测长静止段
        long_stationary_segments = self._find_long_stationary_segments(is_stationary)

        # 合并动作段，直到遇到长静止段
        action_segments = self._merge_until_long_stationary(long_stationary_segments)

        return motion_energy, action_segments
    
    def _calculate_threshold_T(self, motion_energy):
        """
        动态计算阈值 threshold_T
        :param motion_energy: 每一帧的运动能量
        :return: 动态计算的阈值
        """
        threshold_T_values = []

        # 滑动窗口遍历运动能量序列
        for i in range(len(motion_energy) - self.long_stationary_threshold + 1):
            window = motion_energy[i:i + self.long_stationary_threshold]
            # 计算窗口内最大值与最小值的差值
            frame_diff = np.max(window) - np.min(window)

            # 如果差值小于 max_frame_diff，则认为是有效窗口
            if frame_diff < self.max_frame_diff:
                # 计算窗口内的较高值（例如 90% 分位数）
                threshold_T_values.append(np.percentile(window, 100))

        # 如果没有有效窗口，则使用全局 90% 分位数作为阈值
        if len(threshold_T_values) == 0:
            threshold_T = np.percentile(motion_energy, 90)
        else:
            # 取所有有效窗口较高值的平均值作为全局阈值
            threshold_T = np.max(threshold_T_values)

        return threshold_T
    
    def _smooth_frames(self, skeleton_frames):
        """
        对骨骼点坐标进行平滑处理
        :param skeleton_frames: 骨骼点坐标序列
        :return: 平滑后的骨骼点坐标序列
        """
        smoothed_frames = np.zeros_like(skeleton_frames)
        for joint in range(skeleton_frames.shape[1]):
            for dim in range(3):  # x, y, z 三个维度
                smoothed_frames[:, joint, dim] = gaussian_filter1d(
                    skeleton_frames[:, joint, dim], sigma=self.smooth_sigma
                )
        return smoothed_frames

    def _calculate_motion_energy(self, skeleton_frames):
        """
        计算每一帧的整体运动能量
        :param skeleton_frames: 骨骼点坐标序列
        :return: 每一帧的运动能量，形状为 (N,)
        """
        # 计算每一帧的帧间差分，每一个点的x, y, z坐标合并
        velocity = np.linalg.norm(np.diff(skeleton_frames, axis=0), axis=2)  # 形状为 (N-1, M)
        # 计算每一帧的整体运动能量
        motion_energy = np.sum(velocity, axis=1)  # 形状为 (N-1,)
        # 在第一帧前补0，保持与帧数一致
        motion_energy = np.insert(motion_energy, 0, 0)
        return motion_energy

    def _split_actions(self, is_stationary):
        """
        根据静止帧分割动作序列
        :param is_stationary: 每一帧是否为静止帧的布尔数组
        :return: 分割后的动作段列表
        """
        action_segments = []
        current_segment = []

        for frame_idx, stationary in enumerate(is_stationary):
            if stationary:
                # 检测到静止帧，结束当前动作段
                if current_segment:
                    action_segments.append(current_segment)
                    current_segment = []
            else:
                # 继续当前动作段
                current_segment.append(frame_idx)

        # 添加最后一个动作段
        if current_segment:
            action_segments.append(current_segment)

        return action_segments

    def _find_long_stationary_segments(self, is_stationary):
        """
        检测长静止段（超过 long_stationary_threshold 帧的静止段）
        :param is_stationary: 每一帧是否为静止帧的布尔数组
        :return: 长静止段的起始和结束帧索引列表
        """
        long_stationary_segments = []
        start = None

        for frame_idx, stationary in enumerate(is_stationary):
            if stationary and start is None:
                start = frame_idx  # 记录长静止段的起始帧
            elif not stationary and start is not None:
                if frame_idx - start >= self.long_stationary_threshold:
                    long_stationary_segments.append((start, frame_idx - 1))
                start = None  # 重置起始帧

        # 检查最后一个静止段
        if start is not None and len(is_stationary) - start >= self.long_stationary_threshold:
            long_stationary_segments.append((start, len(is_stationary) - 1))

        return long_stationary_segments

    def _merge_until_long_stationary(self, long_stationary_segments):
        """
        按照长静止段的 frameId 分割动作段，并排除长静止帧
        :param action_segments: 初步分割的动作段列表
        :param long_stationary_segments: 长静止段的起始和结束帧索引列表
        :return: 分割后的动作段列表
        """
        # 提取长静止段的起始帧和终止帧
        long_stationary_frames = []
        for start, end in long_stationary_segments:
            long_stationary_frames.append((start, end))

        # 添加边界帧
        long_stationary_frames.append((-1, -1))  # 第一帧的前一帧
        long_stationary_frames.append((self.frame_Count, self.frame_Count))  # 最后一帧的下一帧

        # 按起始帧排序
        long_stationary_frames.sort()

        # 分割动作段
        processed_segments = []
        for i in range(len(long_stationary_frames) - 1):
            # 当前长静止段的结束帧
            current_end = long_stationary_frames[i][1]
            # 下一个长静止段的起始帧
            next_start = long_stationary_frames[i + 1][0]

            # 动作段的起始帧为当前长静止段的结束帧 + 1
            # 动作段的终止帧为下一个长静止段的起始帧 - 1
            start = current_end + 1
            end = next_start - 1

            # 确保动作段有效
            if start <= end:
                processed_segments.append((start, end))

        return processed_segments


def plot_motion_energy(motion_energy, action_segments, threshold_T, long_stationary_segments, save_path="plots/motion_energy_plot.png"):
    """
    绘制运动能量曲线并标注动作分割线，并将图像保存到指定路径。
    
    :param motion_energy: 每一帧的运动能量
    :param action_segments: 分割后的动作段列表
    :param threshold_T: 静止状态的阈值
    :param long_stationary_segments: 长静止段的起始和结束帧索引列表
    :param save_path: 保存图片的路径（默认为 "plots/motion_energy_plot.png"）
    """
    plt.figure(figsize=(12, 6))

    # 绘制运动能量曲线
    plt.plot(motion_energy, label="Motion Energy", color="blue")

    # 绘制静止状态阈值线
    plt.axhline(y=threshold_T, color="red", linestyle="--", label="Threshold (T)")

    # 标注动作分割线
    for segment in action_segments:
        start_frame = segment[0]
        end_frame = segment[-1]
        plt.axvline(x=start_frame, color="green", linestyle="--", alpha=0.5)
        plt.axvline(x=end_frame, color="green", linestyle="--", alpha=0.5)
        plt.fill_betweenx(
            y=[0, max(motion_energy)],
            x1=start_frame,
            x2=end_frame,
            color="green",
            alpha=0.1,
            label="Action Segment" if segment == action_segments[0] else "",
        )

    # 标注长静止段
    for start, end in long_stationary_segments:
        plt.axvspan(start, end, color="gray", alpha=0.3, label="Long Stationary Segment" if start == long_stationary_segments[0][0] else "")

    # 添加图例和标签
    plt.xlabel("Frame Index")
    plt.ylabel("Motion Energy")
    plt.title("Motion Energy and Action Segmentation")
    plt.legend()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path, dpi=300)
    print(f"运动能量图已保存至: {save_path}")


def save_action_segments(segments, filename="action_segments.txt"):
    """
    保存动作分割点到文件
    """
    with open(filename, "w") as f:
        for start, end in segments:
            f.write(f"{start},{end}\n")
    print(f"动作分割信息已保存至 {filename}")


def save_skeleton_segments(skeleton_frames, action_segments, save_dir="segmented_skeletons"):
    """
    按照动作分割点保存骨骼数据到不同文件夹
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (start, end) in enumerate(action_segments):
        segment_data = skeleton_frames[start:end + 1]
        segment_folder = os.path.join(save_dir, f"action_{i + 1}")
        os.makedirs(segment_folder, exist_ok=True)
        np.save(os.path.join(segment_folder, "skeleton.npy"), segment_data)
        print(f"动作 {i + 1}（帧 {start}-{end}）已保存至 {segment_folder}/skeleton.npy")

# 示例用法
if __name__ == "__main__":
    # 假设有200帧骨骼点数据，每帧有10个关节点，每个关节点有(x, y, z)坐标
    
    skeleton_frames = np.load("elfi-splited/skeleton_npy8/body_skeleton.npy", allow_pickle=True)  # 随机生成示例数据
    print(skeleton_frames.shape)
    # 创建动作分割器
    segmenter = ActionSegmenter(frame_Count=len(skeleton_frames), min_segment_length=60, long_stationary_threshold=40, smooth_sigma=1.0)

    # 分割动作序列
    motion_energy, action_segments = segmenter.segment_actions(skeleton_frames)

    # 检测长静止段
    is_stationary = motion_energy < segmenter.threshold_T
    long_stationary_segments = segmenter._find_long_stationary_segments(is_stationary)

    # 保存分割点
    save_action_segments(action_segments)

    # 额外：将骨骼数据按动作段保存
    save_skeleton_segments(skeleton_frames, action_segments)

    # 绘制运动能量曲线和动作分割线
    plot_motion_energy(motion_energy, action_segments, segmenter.threshold_T, long_stationary_segments, save_path="results/motion_energy_analysis.png")

