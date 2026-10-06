import streamlit as st 
import pandas as pd
import joblib

# Load Model
model = joblib.load(
    "C:/Users/Admin/Downloads/Online Food Delivery Prediction(ML)/Online_Food_Delivery_model.pkt"
)

# Load Columnns
columns = joblib.load(
    "C:/Users/Admin/Downloads/Online Food Delivery Prediction(ML)/x_columns.pkt"
)

# Title
st.title("🍔 Online Food Delivery Prediction")

# Add Image
st.image(
    "C:/Users/Admin/Downloads/Online Food Delivery Prediction(ML)/food image.png",
    caption="Online Food Delivery",
    use_container_width=True
)
st.write("Enter Customer details to Predict food delivery preference")

# Inputs
age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)


occupation = st.selectbox(
    "Occupation",
    ["Student","Employee","Self Employed","House wife"]
)

monthly_income=st.selectbox(
    "Monthly Income",
    [
        "No Income",
        "Below Rs.10000",
        "10001 to 25000",
        "25001 to 50000",
        "More than 50000"
    ]
)

family_size = st.number_input(
    "Family Size",
    min_value=1,
    max_value=20,
    value=4
)

feedback = st.selectbox(
    "Feedback",
    ["Positive","Negative"]
)

# Prediction
if st.button("🔮 Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Occupation": [occupation],
        "Monthly Income": [monthly_income],
        "Family size": [family_size],
        "Feedback": [feedback]
    })

# Encoding

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

# Prediction

    prediction = model.predict(input_data)

    if prediction == 1:
        st.success(
            "🍔 Customer is likely to order online food!"
        )
    else:
        st.error(
            "❌ Customer is unlikely to order online food."
        )

