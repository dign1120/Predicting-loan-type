import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

CSV_PATH = "backend/migration_population_balanced.csv"
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
    
    def normalize_test_data(self, test_data): # 실제 feature 1행으로 이루어진 데이터 테스트 목적
        normalized_data = pd.DataFrame(test_data.reshape(1, -1), columns=self.feature_names)
        normalized_data = self.scaler.transform(normalized_data)

        return torch.tensor(normalized_data, dtype=torch.float)
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 24) # 3 features
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 4) # 4 classes
    def forward(self, x):
        x = F.relu(self.fc1(x.to(torch.float)))
        x = F.relu(self.fc2(x.to(torch.float)))
        x = self.fc3(x)
        return x
    


# dataset = CustomDataset(CSV_PATH)
# test_dataset = torch.load("backend/model/test_dataset.pth")   # 이관받은 테스트 셋 사용

# test_loader = DataLoader(
#     dataset = test_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True,
#     )

# model = Net()
# model.load_state_dict(torch.load('backend/model/migration_model.pt')) # 이관받은 모델 사용
# model.to(DEVICE)


### confusion matrix plot for testloader
def plot_confusion_matrix(model, test_loader):
    y_pred = []
    y_true = []

    classes = ( 'IDA',
                'IBRD',
                'Blend',
                'Aggregates' )

    for data,target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        output = output.max(1, keepdim=True)[1].data.cpu().numpy()
        output = output.flatten()
        y_pred.extend(output)
        
        targets = target.data.cpu().numpy()
        y_true.extend(targets)

    cf_metrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cf_metrix / np.sum(cf_metrix, axis=1)[:, None], index = [i for i in classes],  columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True, cmap = "gist_heat")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.savefig('confusion_matrix.png')
    return None


### P-R curve each class
def plot_precision_recall(model, test_loader):
    average_precision_values = []

    
    model.eval()
    y_true = []
    y_scores = []
    classes = ( 'IDA',
                'IBRD',
                'Blend',
                'Aggregates' )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # Calculate probabilities
            softmax_output = F.softmax(output, dim=1)
            y_scores.extend(softmax_output.cpu().numpy())
            targets = target.data.cpu().numpy()
            y_true.extend(targets)

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    print(y_scores)

    
    # Plot precision-recall curves
    plt.figure(figsize=(12, 7))
    ap = 0.0
    for i in range(y_scores.shape[1]):
        precision, recall, threshold = precision_recall_curve(y_true == i, y_scores[:, i])
        # print(y_true == i)
        # print(threshold)
        # print(y_scores[:, i])
        plt.plot(recall, precision, label=f'Class {i} : {classes[i]}')
        average_precision_values.append(average_precision_score(y_true == i, y_scores[:, i]))

    mAP = np.mean(average_precision_values)
    print(f"Mean Average Precision (mAP): {mAP:.2f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('PR_Curve.png')
    return None


### accuracy evaluate
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
            
            pred = output.max(1, keepdim=True)[1]       # 가장 큰 값을 가진 index가 모델의 예측
            correct += pred.eq(target.view_as(pred)).sum().item()       # 예측과 정답을 비교하여 일치할 경우 correct += 1
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy


### predict_lendingType
def predict_class(model, data, dataset, device):
    classes = { 'IDA' : 0,
                'IBRD': 1,
                'Blend': 2,
                'Aggregates': 3 }
    
    specific_normalized = dataset.normalize_test_data(data).to(device)

    # Making predictions
    model.eval()
    with torch.no_grad():
        output = model(specific_normalized)

    # Converting logits to probabilities using softmax
    probabilities = F.softmax(output, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    index_to_class = { v: k for k, v in classes.items() } # 역변환
    predicted_class = index_to_class[predicted_label]
    print("Predicted lendingType :", predicted_class)
    return predicted_class


#plot_confusion_matrix(model, test_loader)               # plot confusion-matrix
#plot_precision_recall(model, test_loader)               # plot PR-Curve each classes
#evaluate(model, test_loader)                            # evaluate with accuracy


# specific data test ( population, net_migration, incomeLevelRank)
#specific_data = np.array([80000, -3000, 1])             
#predict_class(model, specific_data, dataset, DEVICE)
