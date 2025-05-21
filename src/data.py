import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from graph import keypoints

# Tách thông tin từ image_name ở dataset
def extract_info(image_path):
    parts = image_path.strip("./").split("/")
    folder = parts[0]  # '001-bg-01-000'
    filename = parts[1]  # '000001.jpg'
    subject_id, condition, condition_index, angle = folder.split("-")
    frame_index = int(filename.replace(".jpg", ""))
    return int(subject_id), condition, int(condition_index), int(angle), frame_index

class ProcessDataset(Dataset):
    def __init__(self, csv_path, num_frames=60, transform=None, seed=1):
        self.df = pd.read_csv(csv_path)
        self.num_frames = num_frames
        self.transform = transform
        self.seed = seed

        extracted_info = self.df["image_name"].apply(extract_info)
        self.df["subject_id"] = extracted_info.apply(lambda x: x[0])
        self.df["condition"] = extracted_info.apply(lambda x: x[1])
        self.df["condition_index"] = extracted_info.apply(lambda x: x[2])
        self.df["angle"] = extracted_info.apply(lambda x: x[3])
        self.df["frame_index"] = extracted_info.apply(lambda x: x[4])

        self.information = []
        if self.seed is not None:
            np.random.seed(self.seed)
            
        grouped = self.df.groupby(["subject_id", "condition", "condition_index", "angle"])

        for (subject_id, condition, condition_index, angle), group_df in grouped:
            group_df_sorted = group_df.sort_values("frame_index")

            if len(group_df_sorted) >= self.num_frames:
              # start = np.random.randint(0, len(group_df_sorted) - self.num_frames + 1)
              # group_df_sorted = group_df_sorted[start:start+self.num_frames]
              group_df_sorted = group_df_sorted.iloc[:self.num_frames]
              
              node_features = np.zeros((self.num_frames, 17, 3))
              for i, (_, row) in enumerate(group_df_sorted.iterrows()):
                  for j, kp in enumerate(keypoints):
                      node_features[i, j, 0] = row[f"{kp}_x"]
                      node_features[i, j, 1] = row[f"{kp}_y"]
                      node_features[i, j, 2] = row[f"{kp}_conf"]
              self.information.append({
              "subject_id": subject_id,
              "condition": condition,
              "condition_index": condition_index,
              "angle": angle,
              "node_features": node_features
              })
            else:
              continue
            # else:
            #     node_features = np.zeros((len(group_df_sorted), 17, 3))
            #     for i, (_, row) in enumerate(group_df_sorted.iterrows()):
            #         for j, kp in enumerate(keypoints):
            #             node_features[i, j, 0] = row[f"{kp}_x"]
            #             node_features[i, j, 1] = row[f"{kp}_y"]
            #             node_features[i, j, 2] = row[f"{kp}_conf"]


    def __len__(self):
        return len(self.information)

    def __getitem__(self, idx):
        data = self.information[idx]
        node_features = data["node_features"]

        if self.transform:
            node_features = self.transform(node_features)

        return {
            "subject_id": data["subject_id"],
            "condition": data["condition"],
            "condition_index": data["condition_index"],
            "angle": data["angle"],
            "node_features": torch.tensor(node_features, dtype=torch.float32)
        }


