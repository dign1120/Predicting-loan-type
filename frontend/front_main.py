import requests
import streamlit as st

st.title("Loan type prediction API test")

popul_option = st.text_input(label = "How population in country?", value = "input population", key = "population")   #input text : population
mig_option = st.text_input(label = "How net_migration in country?", value = "input net_migration", key = "net_migration")   #input text : net_migration
incomeLevel_option = st.selectbox(label = "What is the income level in your country?", 
                                    options= ('Low income','Lower middle income','Upper middle income','High income','Aggregates'))   #selectbox for incomeLevel


incomeMapper = { "Low income": 1,
                    "Lower middle income" : 2,
                    "Upper middle income" : 3,
                    "High income" : 4,
                    "Aggregates": 5
                    }


if st.button("Predict"):
    url = "http://127.0.0.1:8000/predict_loan"
    

    body = { "population" : popul_option,
            "net_migration" : mig_option,
            "incomeLevel" : incomeMapper[incomeLevel_option]
            }
    
    res = requests.post(url, json = body)
    st.write(str(res.json))
    st.write(str(res.text))




