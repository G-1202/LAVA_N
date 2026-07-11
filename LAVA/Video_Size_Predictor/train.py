import numpy as np
import pandas as pd
import os
cuda = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import torch.nn.functional as F
from tqdm import tqdm
from val import val
from network import V_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

NUM = 20000

INPUT_STEPS = 8
FEATURES = 5
OUTPUT_STEPS = 1
OUTPUT = 176

class VideoDataset(Dataset):
    def __init__(self, df, num, input_steps, OUTPUT_STEPS, len, seed=42):
        self.df = df
        self.num = num
        self.input_steps = input_steps
        self.output_steps = OUTPUT_STEPS
        self.len = len
        self.seed = seed

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        """
            Returns the idx-th training sample.

            Return format: (Temporal_features, Size_labels)
            - Temporal_features: Tensor with shape [input_steps, 7]
            - Size_labels: Tensor with shape [176]
        """
        # Core algorithm implementation details are withheld
        # Complete code will be released upon paper acceptance
        X1, S = self._generate_sample(idx)
        return X1, S

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.mode = mode
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if (self.mode == 'min' and current_value < self.best_value - self.min_delta) or \
                (self.mode == 'max' and current_value > self.best_value + self.min_delta):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

if __name__ == '__main__':
    os.makedirs('Results_v', exist_ok=True)
    log_file = open('Results_v/training_log.txt', 'a')
    log_header = (
        "Epoch | Train Loss | Val Loss \n"
    )
    log_file.write(log_header)
    log_file.flush()

    # Create your own dataset
    df1 = pd.read_hdf(f'dataset/train.h5', 'encoding_data')
    dataset_train = VideoDataset(df1, 'train', NUM, INPUT_STEPS, OUTPUT_STEPS)

    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    size_model = V_Model(model_dim=128, output_dim=OUTPUT).to(device)
    size_optimizer = optim.Adam(size_model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(size_optimizer, T_max=200)

    early_stopper = EarlyStopper(patience=30, mode='min')
    num_epochs = 500
    for epoch in range(num_epochs):
        size_model.train()
        size_loss_total = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for X, y in progress:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            size_pred = size_model(X)
            loss_v = F.l1_loss(size_pred, y)

            size_optimizer.zero_grad()
            loss_v.backward()
            size_optimizer.step()
            size_loss_total += loss_v.item()
            progress.set_postfix({'Loss': f'{loss_v.item():.4f}'})

        with torch.no_grad():
            val = val(size_model, cuda)
        log_content = (
            f"[{epoch + 1:3d}/{num_epochs}]"
            f"  {size_loss_total / len(train_loader):.4f}"
            f"  {val:.4f}\n"
        )
        log_file.write(log_content)
        log_file.flush()
        if (epoch + 1) % 10 == 0:
            torch.save(size_model.state_dict(), f'model/{epoch + 1:03d}_size.pth')
        if early_stopper(val):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    log_file.close()


