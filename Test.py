from CNN.PytorchDataset import *
from CNN.Model import *
import os
import multiprocessing

NEIGHBOUR = 9
RADIUS = int((NEIGHBOUR - 1) / 2)
CHANNEL = 10 + 2
OLD_YEAR = str(2000)
NEW_YEAR = str(2000)

OUTPUT_DIR1 = os.path.join("./DATA/CA_DATA/", OLD_YEAR)
OUTPUT_DIR2 = os.path.join("./DATA/CA_DATA/", NEW_YEAR)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    MODEL = "DualNet3"
    model = DualNet3().to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR1, str(NEIGHBOUR), MODEL + "_checkpoint.pt")))
    test_dataset = CustomDataset(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "test.csv"))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=multiprocessing.cpu_count())

    time_start = time.time()
    potentialMap, _ = test(test_loader, model, nn.BCELoss(), device)
    print("Time:", time.time() - time_start)
    potentialMap = np.array(potentialMap)
    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), OLD_YEAR + "pre"), potentialMap)
