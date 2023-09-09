from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import pandas as pd
import numpy as np
import torch

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, indexDir):
        self.indexDir = pd.read_csv(indexDir, header=None)

    def __len__(self):
        return len(self.indexDir)

    def __getitem__(self, idx):
        # time_start = time.time()
        npy_path = self.indexDir.iloc[idx, 0]
        data = np.load(npy_path)
        label = self.indexDir.iloc[idx, 1]

        data = torch.tensor(data.astype(np.float32))
        data = data.permute(2, 0, 1)
        label = torch.tensor(label.astype(np.float32))
        # print(idx, time.time() - time_start)
        return data, label

