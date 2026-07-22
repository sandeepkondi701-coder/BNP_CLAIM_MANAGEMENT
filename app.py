import streamlit as st

from pipeline.prediction_pipeline import PredictPipeline, CustomData

st.title("BNP CLAIM PREDICTION")

customer_age = st.number_input(
    "customer_age",
    min_value = 18,
    max_value = 100,
    value = 25
)


annual_premium = st.number_input(
    "annual_premium",
    min_value = 1000,
    value= 10000
)

claim_type = st.selectbox(
    "claim_type",
    [
        "Health",
        "Life",
        "Property",
        "Travel"
    ]
)

predict_button = st.button(
    "predict"
)

if predict_button:

    data = CustomData(
        customer_age = customer_age,
        annual_premium = annual_premium,
        claim_type = claim_type
    )

    df = (
        data.get_data_as_dataframe()
    )

    predictor = (
        PredictPipeline()
    )

    prediction = (
        predictor.predict(
            df
        )
    )

    if prediction[0] ==1:
        st.success(
            "claim will be processed"
        )

    else:

        st.error(
            "claim will not be processed"
        )

