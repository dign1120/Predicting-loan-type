from fastapi import FastAPI
from pydantic import BaseModel

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

@app.post("/predict_loan")
def test_api(item : LoanPrdictionTest):
    print({"population" : item.population, 
            "net_migration" : item.net_migration, 
            "incomeLevel" : item.incomeLevel})
    
    return { "population" : item.population, 
            "net_migration" : item.net_migration, 
            "incomeLevel" : item.incomeLevel }
