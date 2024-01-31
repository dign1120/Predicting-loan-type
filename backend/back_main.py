import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import uvicorn

from custom_dataset import CustomDataset, predict_class
from fastapi import FastAPI
from pydantic import BaseModel

CSV_PATH = "backend/migration_population_balanced.csv"
BATCH_SIZE = 128
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

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

app = FastAPI()

class LoanPrdictionTest(BaseModel):
    population : str
    net_migration : str
    incomeLevel : str

@app.get("/")
def root_api():
    return {"message" : "Hello Loan predict!!"}

@app.get("/test")
def test():
    return {"population" : 0,
            "net_migration" : 0,
            "incomeLevel" : "test"}

@app.post("/api_test")
def test_api(item : LoanPrdictionTest):

    return  { "population" : item.population, 
            "net_migration" : item.net_migration, 
            "incomeLevel" : item.incomeLevel }



@app.post("/predict_loan")
def predict_loan(item : LoanPrdictionTest):

    dataset = CustomDataset(CSV_PATH)

    model = Net()
    model.load_state_dict(torch.load('backend/model/migration_model.pt')) # 이관받은 모델 사용
    model.to(DEVICE)

    data_json = { 
                "population" : item.population, 
                "net_migration" : item.net_migration, 
                "incomeLevel" : item.incomeLevel
                }
    
    numpy_data = np.array([data_json['population'], data_json["net_migration"], data_json["incomeLevel"]])
    
    predict_value = predict_class(model, numpy_data, dataset, DEVICE)

    return { "population" : item.population, 
            "net_migration" : item.net_migration, 
            "incomeLevel" : item.incomeLevel,
            "predicted lending type": str(predict_value)
            }

if __name__ == "__main__":
    uvicorn.run("back_main:app", host='0.0.0.0', port=8000, reload=True)

