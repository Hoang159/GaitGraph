import numpy as np

class MultiInput(object):
    def __init__(self, connect_joint, enabled=False):
        self.connect_joint = connect_joint
        self.enabled = enabled

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        if not self.enabled:
            return data[np.newaxis, ...]

        C, T, V = data.shape
        data_new = np.zeros((3, C * 2, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[0, C:, :, i] = data[:, :, i] - data[:, :, 1]
        # Velocity
        for i in range(T - 2):
            data_new[1, :C, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[1, C:, i, :] = data[:, i + 2, :] - data[:, i, :]
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[2, :C, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(data_new[2, i, :, :], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            data_new[2, C, :, :] = np.arccos(data_new[2, i, :, :] / bone_length)

        return data_new


class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class InterpolateFrames:
    # Tăng số khung bằng cách nội suy ngẫu nhiên giữa các khung kế tiếp,
    # Trả về đúng 60 khung bằng cách cắt bớt hoặc thêm padding.

    def __init__(self, probability=0.1, target_frames=60):
        self.probability = probability
        self.target_frames = target_frames

    def __call__(self, data):

        # data shape: T, V, C (Frames, Joints, Channels)
        T, V, C = data.shape
        interpolated_data = []

        if T >= self.target_frames:
          interpolated_data = data[:self.target_frames]
        else:

          for i in range(T):
              interpolated_data.append(data[i])

              if i == T - 1:
                  break

              # Ngẫu nhiên chèn nội suy, nếu < xác suất mới bỏ qua
              # if np.random.random() <= self.probability:
              #     continue

              x_diff = data[i + 1, :, 0] - data[i, :, 0]
              y_diff = data[i + 1, :, 1] - data[i, :, 1]
              conf_avg = (data[i + 1, :, 2] + data[i, :, 2]) / 2

              interp_x = data[i, :, 0] + x_diff * np.random.normal(0.5, 1)
              interp_y = data[i, :, 1] + y_diff * np.random.normal(0.5, 1)

              interp_frame = np.array([interp_x, interp_y, conf_avg]).transpose()
              interpolated_data.append(interp_frame)

          interpolated_data = np.array(interpolated_data)

          # Resize/pad để đảm bảo đủ target_frames
          current_frames = interpolated_data.shape[0]

          if current_frames < self.target_frames:
              # Lặp lại khung cuối
              pad = np.tile(interpolated_data[-1:], (self.target_frames - current_frames, 1, 1))
              interpolated_data = np.concatenate([interpolated_data, pad], axis=0)

        return interpolated_data


class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0).copy()
        return data
    

class ShuffleSequence(object):
    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, data):
        if self.enabled:
            np.random.shuffle(data)
        return data


class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, 0.25, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)

class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise
    
class TwoNoiseTransform(object):
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]