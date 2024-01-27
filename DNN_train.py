import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
from Early_stopping import EarlyStopping


CSV_PATH = "migration_population_balanced.csv"
EPOCHS = 1000
BATCH_SIZE = 128
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class CustomDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        self.feature_names = ['population', 'net_migration', 'incomeLevelRank']

        scaler = MinMaxScaler()
        scaler.fit(df[self.feature_names])
        scaled_x = scaler.transform(df[self.feature_names])
        df_scaled = pd.DataFrame(scaled_x)
        #shape: (2164, 4)
        self.scaler = scaler
        self.x = df_scaled.iloc[:, :].values
        self.y = df.iloc[:, -1]

    def __len__(self):
        return len(self.x) # 2164
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

dataset = CustomDataset(CSV_PATH)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    )

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 24) # 3개의 독립변수에 대해 4가지 class로 예측
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x.to(torch.float)))
        x = F.relu(self.fc2(x.to(torch.float)))
        x = self.fc3(x)
        return x

model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # optimizer 설정 : 1. SGD, 2: Adagrad, 3 : RMSprop, 4 : Adam

def train(model, train_loader, optimizer):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE) # 학습 데이터를 DEVICE의 메모리로 보냄
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    return loss
    

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# plot train, test loss
def plot_loss(train_loss_list, test_loss_list):
    train_loss_list = [loss.detach() for loss in train_loss_list]
    plt.plot(range(1,stopeed_epoch+1), np.array(train_loss_list),'g', label = 'train_loss')
    plt.plot(range(1,stopeed_epoch+1), np.array(test_loss_list),'b', label = 'test_loss')
    plt.axvline(stopeed_epoch, 0, 1, color='r', linestyle='--', linewidth=2, label = "early stopping")
    plt.annotate(f"{test_loss_list[stopeed_epoch -1]:.4f}", (stopeed_epoch -1, test_loss_list[stopeed_epoch -1]), textcoords="offset points", xytext=(-15, 5), ha='center', fontsize=10, color='b', weight='bold')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title("optimizer : Adam")
    plt.savefig("error_plot_Adam.png")
    plt.show()
    return None


early_stopping = EarlyStopping(patience = 3, verbose = True)
train_loss_list = []
test_loss_list = []
stopeed_epoch = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, optimizer)
    train_loss_list.append(train_loss.cpu()) # tensor -> numpy
    test_loss, test_accuracy = evaluate(model, test_loader)
    test_loss_list.append(test_loss)
    print(f'test loss : {test_loss}')

    early_stopping(test_loss, model)
    stopeed_epoch = epoch
    if early_stopping.early_stop:
        break

plot_loss(train_loss_list, test_loss_list)
torch.save(test_dataset, 'test_dataset.pth') # 데이터셋 이관
torch.save(model.state_dict(), 'migration_model.pt') # 모델 이관

