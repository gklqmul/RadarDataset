import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ActionDataset(Dataset):
    def __init__(self, root_dir, task='action_classification', padding_mode='zero',sampling_mode='random', max_points=32, max_frames=250, scale_factor=1.0, test_size=0.2, val_size=0.1, stack_func=None, normalize=False, standardize=False):
        self.root_dir = root_dir
        self.task = task
        self.padding_mode = padding_mode
        self.sampling_mode = sampling_mode
        self.max_points = max_points
        self.max_frames = max_frames
        self.scale_factor = scale_factor
        self.stack_func = stack_func
        self.normalize = normalize
        self.standardize = standardize

        self.data = []

        for env in ['env1']:
            env_path = os.path.join(root_dir, env, 'subjects')
            if not os.path.exists(env_path):
                continue

            for subject in os.listdir(env_path):
                aligned_path = os.path.join(env_path, subject, 'aligned')
                if not os.path.exists(aligned_path):
                    continue

                for action_folder in os.listdir(aligned_path):
                    action_path = os.path.join(aligned_path, action_folder)
                    if not os.path.isdir(action_path):
                        continue

                    radar_file = [f for f in os.listdir(action_path) if f.endswith('.h5')]
                    skeleton_file = [f for f in os.listdir(action_path) if f.endswith('.npy')]

                    if radar_file and skeleton_file:
                        radar_path = os.path.join(action_path, radar_file[0])
                        skeleton_path = os.path.join(action_path, skeleton_file[0])
                        self.data.append((radar_path, skeleton_path))
        print(len(self.data))  # 看看是不是0

        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)

        self.datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def scale_data(self, data, data_type='radar'):
        if data_type == 'radar':
            data = data * 1000
        elif self.normalize:
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data = (data - data_min) / (data_max - data_min + 1e-8)
        elif self.standardize:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data = (data - data_mean) / (data_std + 1e-8)
        return data

    def get_dataset(self, split='train'):
        dataset = self.datasets.get(split, [])
        return [self.process_sample(radar_path, skeleton_path) for radar_path, skeleton_path in dataset]

    def process_sample(self, radar_path, skeleton_path):
        with h5py.File(radar_path, 'r') as f:
            frames_group = f['frames']
            radar_data = []

            for frame_name in frames_group:
                frame_ds = frames_group[frame_name]
                frame_array = np.array(frame_ds)
                processed_frame = self.pad_or_trim_points(self.scale_data(frame_array, 'radar'))
                processed_frame = processed_frame[:, [5, 6, 1, 3]]  # 选取四个维度
                radar_data.append(processed_frame)  # 转置

        radar_data = np.array(radar_data)
        radar_data = self.pad_or_trim_frames(radar_data)

        skeleton_data = np.load(skeleton_path)
        skeleton_data = self.pad_or_trim_frames(skeleton_data)

        frame_count = min(len(radar_data), len(skeleton_data))
        radar_data = radar_data[:frame_count]
        skeleton_data = skeleton_data[:frame_count]

        action_label = int(radar_path.split('action')[-1][:2])
        subject_label = int(radar_path.split('subject')[-1][:2])

        sample = {
            'radar': radar_data,
            'skeleton': skeleton_data
        }

        if self.task == 'action_classification':
            sample['label'] = action_label
        elif self.task == 'subject_identification':
            sample['label'] = subject_label
        elif self.task == 'keypoint_estimation':
            sample['label'] = skeleton_data

        return sample

    def pad_or_trim_points(self, frame):
        num_points, point_attributes = frame.shape

        if num_points < self.max_points:
            if self.padding_mode == 'zero':
                padding = np.zeros((self.max_points - num_points, point_attributes))
            elif self.padding_mode == 'repeat':
                if num_points == 0:
                    raise ValueError("Cannot repeat last point from an empty frame.")
                padding = np.tile(frame[-1:], (self.max_points - num_points, 1))
            else:
                raise ValueError(f"Unsupported padding mode: {self.padding_mode}")
            return np.concatenate([frame, padding], axis=0)

        if num_points > self.max_points:
            if self.sampling_mode == 'truncate':
                return frame[:self.max_points]
            elif self.sampling_mode == 'random':
                indices = np.random.choice(num_points, self.max_points, replace=False)
                return frame[indices]  

        return frame 


    def pad_or_trim_frames(self, data):
        num_frames = data.shape[0]
        if num_frames < self.max_frames:
            if self.padding_mode == 'zero':
                padding = np.zeros((self.max_frames - num_frames, *data.shape[1:]))
            elif self.padding_mode == 'repeat':
                if num_frames == 0:
                    raise ValueError("Cannot repeat last frame from an empty sequence.")
                padding = np.tile(data[-1:], (self.max_frames - num_frames, 1, 1))
            else:
                raise ValueError(f"Unsupported padding mode: {self.padding_mode}")
            return np.concatenate([data, padding], axis=0)
        return data[:self.max_frames]


if __name__ == "__main__":
    dataset = ActionDataset(root_dir='dataset', task='action_classification', scale_factor=1.5)
    train_data = dataset.get_dataset('train')
    val_data = dataset.get_dataset('val')
    test_data = dataset.get_dataset('test')
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"First radar frame shape: {train_data[0]['radar'].shape}")
    print(f"First kinect frame first point: {train_data[0]['skeleton'].shape}")
    print(f"First kinect frame first point: {train_data[0]['label']}")



