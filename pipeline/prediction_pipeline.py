import os

import sys

from src.utils import load_object

from src.exception import CustomException


class PredictPipeline:

    def __init__(self):
        pass

    def predict(
            self,
            features):
        
        try:

            model = load_object(
                "artifacts/model.pkl"
            )

            preprocessor = load_object(
                "artifacts/preprocessor.pkl"
            )

            datascaled =(
                preprocessor.
                transform(
                    features
                )

            )

            prediction = (
                model.predict(
                    datascaled
                )
            )

            return prediction
        
        except Exception as e:

            raise CustomException(
                e,
                sys
            )
        
import pandas as pd

class CustomData:

    def __init__(self,
                 customer_age,
                 claim_type,
                 annual_premium):
       
       self.customer_age = customer_age
       self.claim_type = claim_type
       self.annual_premium = annual_premium
        
    def get_data_as_dataframe(self):

        customdata_input = {

            "customer_age" : [self.customer_age],
            "claim_type" : [self.claim_type], 
            "annual_premium" : [self.annual_premium]

        }

        return pd.DataFrame(
            customdata_input
        )
        

        


        