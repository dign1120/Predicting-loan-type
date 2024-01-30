import requests
import streamlit as st

image_url = "https://cdn-icons-png.flaticon.com/512/2909/2909523.png"

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <h1 style="flex: 1; margin: 0; vertical-align: middle;">Loan type prediction</h1>
        <div style="text-align: center;">
            <img src="{image_url}" alt="Image" style="width: 50px; height: 50px; object-fit: cover; border-radius: 10px;">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Add conditional formatting for input fields
popul_option = st.number_input(
    label="How population in country?",
    value=None,
    step=1,  # Set the step size to 1
    key="population",
    format="%d",  # Format as integer
    help="Required field. Please enter a numeric value in units of days.",
)
mig_option = st.number_input(
    label="How net_migration in country?",
    value=None,
    step=1,  # Set the step size to 1
    key="net_migration",
    format="%d",  # Format as integer
    help="Required field. Please enter a numeric value in units of days.",
)
incomeLevel_option = st.selectbox(
    label="What is the income level in your country?",
    options=('Low income', 'Lower middle income', 'Upper middle income', 'High income', 'Aggregates'),
    help="Select the income level.",
)  # selectbox for incomeLevel


incomeMapper = {"Low income": 1,
                "Lower middle income": 2,
                "Upper middle income": 3,
                "High income": 4,
                "Aggregates": 5}

# Adjust input field styles based on the presence of values
if st.button("Predict"):
    try:
        # Check if required fields are filled and are numeric
        if popul_option is not None and mig_option is not None:
            url = "http://127.0.0.1:8000/predict_loan"
            body = {
                "population": str(popul_option),
                "net_migration": str(mig_option),
                "incomeLevel": str(incomeMapper[incomeLevel_option])
            }
            res = requests.post(url, json=body)

            if res.status_code == 200:
                # Parse the JSON response
                response_data = res.json()
                predicted_lending_type = response_data.get("predicted lending type", "N/A")
                st.markdown(
                    f"<div style='background-color: #82f802; padding: 10px; border-radius: 5px; text-align: center;'>"
                    f"<h3 style='color: white;'>Predicted lending type: {predicted_lending_type}</h3>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.write(f"Failed to get prediction. Status code: {res.status_code}")
        else:
            st.warning("Please fill in the required fields with numeric values.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the server please try again")
