from pipeline.prediction_pipeline import PredictPipeline,CustomData

data = CustomData(

    customer_age=35,

    annual_premium=20000,

    claim_type="Health"


)

df = (
    data.
    get_data_as_dataframe()
)

predictor = (
    PredictPipeline()
)

result = predictor.predict(
    df
)

print(result)