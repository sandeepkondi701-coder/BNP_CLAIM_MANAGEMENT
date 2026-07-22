from fastapi import FastAPI

app = FastAPI()


@app.get("/")

def home():

    return{
        "message":

        "BNP CLAIM PREDICTION API"
    }


from pipeline.prediction_pipeline import PredictPipeline, CustomData


@app.post("/predict")

def predict(
    customer_age : int,
    claim_type : str,
    annual_premium : int):

    data = CustomData(
        customer_age = customer_age,
        claim_type = claim_type,
        annual_premium = annual_premium
    )


    df = (data.
          get_data_as_dataframe())
    

    predictor = (
        PredictPipeline())

    prediction = (
        predictor.predict(
            df
            )
            )
    
    if prediction[0]==1:
        result = "processed"
    
    else:
        result = "Not Processed"

    return{
        "Prediction":
        result
    }

