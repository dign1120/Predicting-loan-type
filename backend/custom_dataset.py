import torch
import pandas as pd

import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


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
    return predicted_class