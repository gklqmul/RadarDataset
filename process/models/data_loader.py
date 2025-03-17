import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ActionDataset(Dataset):
    def __init__(self, root_dir, task='action_classification', padding_mode='zero', max_points=512, scale_factor=1.0, test_size=0.2, val_size=0.1, stack_func=None, normalize=False, standardize=False):
        self.root_dir = root_dir
        self.task = task
        self.padding_mode = padding_mode
        self.max_points = max_points
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

        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)

        self.datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def scale_data(self, data, data_type='radar'):
        if data_type == 'radar':
            data = data * self.scale_factor
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
            radar_data = np.array(f['radar_data'])

        radar_data = radar_data[:, :, [1, 3, 5, 6]]
        radar_data = [frame.T for frame in radar_data]

        if self.stack_func:
            radar_data = self.stack_func(radar_data)

        radar_data = np.array([self.pad_or_trim_points(self.scale_data(frame, 'radar')) for frame in radar_data])

        skeleton_data = np.load(skeleton_path)
        skeleton_data = self.scale_data(skeleton_data, 'skeleton')

        frame_count = min(len(radar_data), skeleton_data.shape[0])
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
        num_points = frame.shape[0]

        if num_points >= self.max_points:
            return frame[:self.max_points]

        if self.padding_mode == 'zero':
            padding = np.zeros((self.max_points - num_points, frame.shape[1]))
        elif self.padding_mode == 'repeat':
            padding = np.tile(frame[-1:], (self.max_points - num_points, 1))
        elif self.padding_mode == 'last_frame':
            padding = np.repeat(frame[-1:], self.max_points - num_points, axis=0)

        return np.concatenate([frame, padding], axis=0)


if __name__ == "__main__":
    dataset = ActionDataset(root_dir='dataset', task='action_classification', scale_factor=1.5)
    train_data = dataset.get_dataset('train')
    print(f"Training samples: {len(train_data)}")
    print(f"Example sample: {train_data[0]}")