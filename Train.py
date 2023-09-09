# -*- coding: UTF-8 -*-
from CNN.PytorchDataset import *
from CNN.EarlyStopping import *
from CNN.Model import *
import os
import multiprocessing

NEIGHBOUR = 9
RADIUS = int((NEIGHBOUR - 1) / 2)
CHANNEL = 10 + 2
OLD_YEAR = str(2000)
OUTPUT_DIR = os.path.join("./DATA/CA_DATA/", OLD_YEAR)

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    MODEL = "DualNet3"
    model = DualNet3().to(device)
    train_dataset = CustomDataset(os.path.join(OUTPUT_DIR, str(NEIGHBOUR), "train.csv"))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=multiprocessing.cpu_count())
    val_dataset = CustomDataset(os.path.join(OUTPUT_DIR, str(NEIGHBOUR), "val.csv"))
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=128, num_workers=multiprocessing.cpu_count())
    test_dataset = CustomDataset(os.path.join(OUTPUT_DIR, str(NEIGHBOUR), "test.csv"))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=multiprocessing.cpu_count())
    print("训练数量", len(train_dataset), "验证数量", len(val_dataset), "测试数量", len(test_dataset))

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    early_stopping = EarlyStopping(os.path.join(OUTPUT_DIR, str(NEIGHBOUR)), MODEL, patience=10, verbose=True)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        time_start = time.time()
        train(train_loader, model, loss_fn, optimizer, device)
        print("Time:", time.time() - time_start)
        valid_loss = val(val_loader, model, loss_fn, device)

        early_stopping(valid_loss, model, t)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Done!")
