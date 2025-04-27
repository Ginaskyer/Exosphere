from sklearn.model_selection import train_test_split as sklearn_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch
import numpy as np

# Function to convert segments to features and labels
# def prepare_data(segments):
#     features = []
#     labels = []

#     for segment in segments:
#         X = segment.drop("Label", axis=1).values

#         # Extract labels
#         y = segment["Label"].values

#         features.append(X)
#         labels.append(y)

#     # Convert to PyTorch tensors
#     features_tensor = torch.FloatTensor(np.array(features))
#     labels_tensor = torch.LongTensor(np.array(labels))

#     return features_tensor, labels_tensor

def cal_entropy(df):
    intervals = df['IAT_norm']
    unique_intervals, counts = np.unique(intervals, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))  # 加 epsilon 防止 log(0)

    return entropy

# def load_dataset(train_segments, test_segments, batch_size=32):
#     """
#     Converts the training and testing segments into PyTorch DataLoader objects.

#     Parameters:
#     train_segments (list): List of dataframes containing training segments
#     test_segments (list): List of dataframes containing testing segments
#     batch_size (int): Batch size for DataLoader

#     Returns:
#     tuple: (train_loader, test_loader) PyTorch DataLoader objects
#     """
#     # Prepare training data
#     X_train, y_train = prepare_data(train_segments)
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # Prepare testing data
#     X_test, y_test = prepare_data(test_segments)
#     test_dataset = TensorDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader

def create_dataset(df, train_test_split, type_value, segment_length=2048, increments=64):
    """
    From dataframe, segment the data, doing increments.
    E.g. if segment_length = 2048, increments = 64, then take 2048/64 = 32 steps.
    From that new segments, take only segments if they have at least 1 packet that has label 1
    Meaning, we exclude segments with only normal segments (all labelled 0).
    Finally, we shuffle them, and split them according to train_test_split ratio
    """
    # Calculate step size (increment)
    step_size = segment_length // increments

    # Create segments
    segments = []
    for i in range(0, len(df) - segment_length + 1, step_size):
        segment = df.iloc[i : i + segment_length]

        # Check if the segment has at least one packet with label 1
        if 1 in segment["Label"].values:  # Assuming the label column is named 'label'
            # entropy = cal_entropy(segment)
            segment = segment.copy()
            # segment.loc[:, "segment_entropy"] = entropy
            segment.loc[:, "Type"] = type_value
            segments.append(segment)

    # If no segments with label 1 were found
    if len(segments) == 0:
        return None, None

    # Shuffle segments
    np.random.shuffle(segments)

    # Split into training and testing sets
    split_idx = int(len(segments) * train_test_split)
    train_segments = segments[:split_idx]
    test_segments = segments[split_idx:]

    return train_segments, test_segments

class Ddosattack(Dataset):
    def __init__(self, dataframe_list, type_number = 5):
        """
        dataframe_list: List[pd.DataFrame], each with columns:
        ["IAT_norm", "PL_norm", "Label", "Type"]
        """
        self.data_list = []
        self.label_list = []

        for segment in dataframe_list:
            for df in segment:
                # 1. 提取输入特征为 (seq_len, 2)
                data = df[["IAT_norm", "PL_norm"]].to_numpy(dtype="float32")
                data_tensor = torch.tensor(data, dtype=torch.float)  # (seq_len, 2)

                # 2. 构造标签为 (seq_len, 1)，每一行为一个 class index
                # 若 label == 0，label_id = 0，否则根据 Type 映射为 1~4
                labels = []
                for i in range(len(df)):
                    label = df.iloc[i]["Label"]
                    type_value = df.iloc[i]["Type"]
                    one_hot = torch.zeros(type_number, dtype=torch.float)
                    if label == 0:
                        one_hot[0] = 1.0
                        labels.append(one_hot)
                    elif 1 <= type_value <= 4:
                        one_hot[int(type_value)] = 1.0
                        labels.append(one_hot)  # 1~4
                    else:
                        raise ValueError(f"Unexpected Type: {type_value} at row {i}")
                
                
                label_tensor = torch.stack(labels, dim=0)
                # print("label_tensor", label_tensor.size())
                
                # print("original", df["Label"])
                # print("data_tensor", data_tensor)
                # print("label_tensor", label_tensor)
                self.data_list.append(data_tensor)
                self.label_list.append(label_tensor)
            

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx]




