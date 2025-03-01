from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import logging
import numpy as np
import os
import joblib  

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()

MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.save"
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing
INCREASE_FACTOR = 1.05  

# Data Model (schema)
class TestReport(BaseModel):
    id: int
    projectName: str
    authors: str
    storyTests: int
    regressionTestsAutomated: int
    regressionTestsManual: int
    totalTestsByApplication: int
    storyPassed: int
    storyFailed: int
    storyUnexecuted: int
    storyBlocked: int
    storySkipped: int
    storyCritical: int
    storyNew: int
    storyUnused: int
    storyBugs: int
    arPassed: int
    arFailed: int
    arUnexecuted: int
    arBlocked: int
    arSkipped: int
    arCritical: int
    arNew: int
    arUnused: int
    arBugs: int
    mrPassed: int
    mrFailed: int
    mrUnexecuted: int
    mrBlocked: int
    mrSkipped: int
    mrCritical: int
    mrNew: int
    mrUnused: int
    mrBugs: int
    createdAt: datetime.datetime


# Sample Data 
sample_data = [
    {"id": 1, "projectName": "ProjectA", "authors": "Alice", "storyTests": 100, "regressionTestsAutomated": 50, "regressionTestsManual": 30, "totalTestsByApplication": 180, "storyPassed": 80, "storyFailed": 5, "storyUnexecuted": 5, "storyBlocked": 2, "storySkipped": 0, "storyCritical": 10, "storyNew": 15, "storyUnused": 3, "storyBugs": 2, "arPassed": 40, "arFailed": 3, "arUnexecuted": 2, "arBlocked": 1, "arSkipped": 0, "arCritical": 5, "arNew": 8, "arUnused": 1, "arBugs": 1, "mrPassed": 25, "mrFailed": 1, "mrUnexecuted": 1, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 3, "mrNew": 5, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-01-01", "arSkipped":0},
    {"id": 2, "projectName": "ProjectA", "authors": "Bob", "storyTests": 105, "regressionTestsAutomated": 52, "regressionTestsManual": 32, "totalTestsByApplication": 189, "storyPassed": 83, "storyFailed": 6, "storyUnexecuted": 4, "storyBlocked": 1, "storySkipped": 1, "storyCritical": 11, "storyNew": 16, "storyUnused": 2, "storyBugs": 1, "arPassed": 42, "arFailed": 2, "arUnexecuted": 1, "arBlocked": 0, "arSkipped": 1, "arCritical": 6, "arNew": 9, "arUnused": 0, "arBugs": 0, "mrPassed": 27, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 4, "mrNew": 6, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-01-08", "arSkipped":1},
    {"id": 3, "projectName": "ProjectA", "authors": "Charlie", "storyTests": 110, "regressionTestsAutomated": 55, "regressionTestsManual": 35, "totalTestsByApplication": 200, "storyPassed": 88, "storyFailed": 7, "storyUnexecuted": 3, "storyBlocked": 0, "storySkipped": 2, "storyCritical": 12, "storyNew": 17, "storyUnused": 1, "storyBugs": 0, "arPassed": 45, "arFailed": 1, "arUnexecuted": 0, "arBlocked": 0, "arSkipped": 2, "arCritical": 7, "arNew": 10, "arUnused": 0, "arBugs": 0, "mrPassed": 30, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 5, "mrNew": 7, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-01-15", "arSkipped":2},
    {"id": 4, "projectName": "ProjectA", "authors": "David", "storyTests": 115, "regressionTestsAutomated": 58, "regressionTestsManual": 38, "totalTestsByApplication": 211, "storyPassed": 93, "storyFailed": 8, "storyUnexecuted": 2, "storyBlocked": 0, "storySkipped": 3, "storyCritical": 13, "storyNew": 18, "storyUnused": 0, "storyBugs": 0, "arPassed": 48, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "arSkipped": 3, "arCritical": 8, "arNew": 11, "arUnused": 0, "arBugs": 0, "mrPassed": 33, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 6, "mrNew": 8, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-01-22", "arSkipped":3},
    {"id": 5, "projectName": "ProjectA", "authors": "Eve", "storyTests": 120, "regressionTestsAutomated": 60, "regressionTestsManual": 40, "totalTestsByApplication": 220, "storyPassed": 96, "storyFailed": 9, "storyUnexecuted": 1, "storyBlocked": 0, "storySkipped": 4, "storyCritical": 14, "storyNew": 19, "storyUnused": 0, "storyBugs": 0, "arPassed": 50, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "arSkipped": 4, "arCritical": 9, "arNew": 12, "arUnused": 0, "arBugs": 0, "mrPassed": 35, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 7, "mrNew": 9, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-01-29", "arSkipped":4},
    {"id": 6, "projectName": "ProjectA", "authors": "Frank", "storyTests": 125, "regressionTestsAutomated": 62, "regressionTestsManual": 42, "totalTestsByApplication": 229, "storyPassed": 99, "storyFailed": 10, "storyUnexecuted": 0, "storyBlocked": 0, "storySkipped": 5, "storyCritical": 15, "storyNew": 20, "storyUnused": 0, "storyBugs": 0, "arPassed": 52, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "storySkipped": 5, "arCritical": 10, "arNew": 13, "arUnused": 0, "arBugs": 0, "mrPassed": 37, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 8, "mrNew": 10, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-02-05", "arSkipped":5},
    {"id": 7, "projectName": "ProjectA", "authors": "Grace", "storyTests": 130, "regressionTestsAutomated": 64, "regressionTestsManual": 44, "totalTestsByApplication": 238, "storyPassed": 102, "storyFailed": 11, "storyUnexecuted": 0, "storyBlocked": 0, "storySkipped": 6, "storyCritical": 16, "storyNew": 21, "storyUnused": 0, "storyBugs": 0, "arPassed": 54, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "storySkipped": 6, "arCritical": 11, "arNew": 14, "arUnused": 0, "arBugs": 0, "mrPassed": 39, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 9, "mrNew": 11, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-02-12", "arSkipped":6},
    {"id": 8, "projectName": "ProjectA", "authors": "Henry", "storyTests": 135, "regressionTestsAutomated": 66, "regressionTestsManual": 46, "totalTestsByApplication": 247, "storyPassed": 105, "storyFailed": 12, "storyUnexecuted": 0, "storyBlocked": 0, "storySkipped": 7, "storyCritical": 17, "storyNew": 22, "storyUnused": 0, "storyBugs": 0, "arPassed": 56, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "storySkipped": 7, "arCritical": 12, "arNew": 15, "arUnused": 0, "arBugs": 0, "mrPassed": 41, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 10, "mrNew": 12, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-02-19", "arSkipped":7},
    {"id": 9, "projectName": "ProjectA", "authors": "Ivy", "storyTests": 140, "regressionTestsAutomated": 68, "regressionTestsManual": 48, "totalTestsByApplication": 256, "storyPassed": 108, "storyFailed": 13, "storyUnexecuted": 0, "storyBlocked": 0, "storySkipped": 8, "storyCritical": 18, "storyNew": 23, "storyUnused": 0, "storyBugs": 0, "arPassed": 58, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "storySkipped": 8, "arCritical": 13, "arNew": 16, "arUnused": 0, "arBugs": 0, "mrPassed": 43, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 11, "mrNew": 13, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-02-26", "arSkipped":8},
    {"id": 10, "projectName": "ProjectA", "authors": "Jack", "storyTests": 145, "regressionTestsAutomated": 70, "regressionTestsManual": 50, "totalTestsByApplication": 265, "storyPassed": 111, "storyFailed": 14, "storyUnexecuted": 0, "storyBlocked": 0, "storySkipped": 9, "storyCritical": 19, "storyNew": 24, "storyUnused": 0, "storyBugs": 0, "arPassed": 60, "arFailed": 0, "arUnexecuted": 0, "arBlocked": 0, "storySkipped": 9, "arCritical": 14, "arNew": 17, "arUnused": 0, "arBugs": 0, "mrPassed": 45, "mrFailed": 0, "mrUnexecuted": 0, "mrBlocked": 0, "mrSkipped": 0, "mrCritical": 12, "mrNew": 14, "mrUnused": 0, "mrBugs": 0, "createdAt": "2024-03-04", "arSkipped":9}
]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Monthly Data Forecasting API!"}


@app.post("/predict")
async def predict_next_month(data: List[TestReport]):
    """
    Predicts the next month's 'totalTestsByApplication' based on provided weekly data using LSTM.
    """
    try:
        logger.debug("Received data: %s", data)
        df = pd.DataFrame([item.dict() for item in data])
        logger.debug("DataFrame created: %s", df.head())
        df['createdAt'] = pd.to_datetime(df['createdAt'])
        logger.debug("createdAt converted to datetime")
        df = df.sort_values(by='createdAt')
        logger.debug("Data sorted by createdAt")

        features = ['storyTests', 'regressionTestsAutomated', 'regressionTestsManual', 'totalTestsByApplication']
        df = df[features + ['createdAt']]  
        df = df.set_index('createdAt')
        df = df.fillna(df.mean())  

        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.debug("Loaded existing scaler from %s", SCALER_PATH)
        else:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)
            joblib.dump(scaler, SCALER_PATH)
            logger.debug("Fitted scaler and saved to %s", SCALER_PATH)

        scaled_data = scaler.transform(df)  

        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), :]
                X.append(a)
                y.append(dataset[i + time_step, 3])
            return np.array(X), np.array(y)

        time_step = 3  
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]) 

        train_size = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # The LSTM model
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            logger.debug("Loaded existing model from %s", MODEL_PATH)
        else:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            logger.debug("LSTM model created")

            model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0) 
            model.save(MODEL_PATH)
            logger.debug("LSTM model fitted and saved to %s", MODEL_PATH)

        # Make prediction
        x_input = scaled_data[-time_step:].reshape((1, time_step, len(features)))
        next_month_scaled_prediction = model.predict(x_input)[0]
        next_month_prediction = scaler.inverse_transform(np.concatenate((np.zeros((1, len(features) - 1)), next_month_scaled_prediction.reshape(1, 1)), axis=1))[0, -1]

        logger.debug("Next month prediction: %s", next_month_prediction)

       
        y_pred = model.predict(X_test)
      
        y_pred = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
        
        y_true = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"MAE: {mae}, RMSE: {rmse}")

        last_month = df.index[-1]
        next_month = last_month + pd.DateOffset(months=1)

        next_month_data = {
            'id': int(data[-1].id + 1),
            'projectName': data[-1].projectName,
            'authors': data[-1].authors,
            'storyTests': int(data[-1].storyTests * INCREASE_FACTOR),  
            'regressionTestsAutomated': int(data[-1].regressionTestsAutomated * INCREASE_FACTOR),
            'regressionTestsManual': int(data[-1].regressionTestsManual * INCREASE_FACTOR),
            'totalTestsByApplication': int(next_month_prediction),
            'storyPassed': int(data[-1].storyPassed * INCREASE_FACTOR),
            'storyFailed': int(data[-1].storyFailed * INCREASE_FACTOR),
            'storyUnexecuted': int(data[-1].storyUnexecuted * (1 + (1 - INCREASE_FACTOR))), 
            'storyBlocked': int(data[-1].storyBlocked * (1 + (INCREASE_FACTOR - 1))),  
            'storySkipped': int(data[-1].storySkipped),
            'storyCritical': int(data[-1].storyCritical * INCREASE_FACTOR),
            'storyNew': int(data[-1].storyNew * INCREASE_FACTOR),
            'storyUnused': int(data[-1].storyUnused * (1 + (1 - INCREASE_FACTOR))),
            'storyBugs': int(data[-1].storyBugs * (1 + (INCREASE_FACTOR - 1))),
            'arPassed': int(data[-1].arPassed * INCREASE_FACTOR),
            'arFailed': int(data[-1].arFailed * INCREASE_FACTOR),
            'arUnexecuted': int(data[-1].arUnexecuted * (1 + (1 - INCREASE_FACTOR))),
            'arBlocked': int(data[-1].arBlocked * (1 + (INCREASE_FACTOR - 1))),
            'arSkipped': int(data[-1].arSkipped),
            'arCritical': int(data[-1].arCritical * INCREASE_FACTOR),
            'arNew': int(data[-1].arNew * INCREASE_FACTOR),
            'arUnused': int(data[-1].arUnused * (1 + (1 - INCREASE_FACTOR))),
            'arBugs': int(data[-1].arBugs * (1 + (INCREASE_FACTOR - 1))),
            'mrPassed': int(data[-1].mrPassed * INCREASE_FACTOR),
            'mrFailed': int(data[-1].mrFailed * INCREASE_FACTOR),
            'mrUnexecuted': int(data[-1].mrUnexecuted * (1 + (1 - INCREASE_FACTOR))),
            'mrBlocked': int(data[-1].mrBlocked * (1 + (INCREASE_FACTOR - 1))),
            'mrSkipped': int(data[-1].mrSkipped),
            'mrCritical': int(data[-1].mrCritical * INCREASE_FACTOR),
            'mrNew': int(data[-1].mrNew * INCREASE_FACTOR),
            'mrUnused': int(data[-1].mrUnused * (1 + (1 - INCREASE_FACTOR))),
            'mrBugs': int(data[-1].mrBugs * (1 + (INCREASE_FACTOR - 1))),
            'createdAt': next_month
        }
        next_month_report = TestReport(**next_month_data)
        logger.debug("Next month report created: %s", next_month_report)

        return {"prediction": int(next_month_prediction),
                "next_month_data": next_month_report.dict(),
                "mae": mae,  
                "rmse": rmse}

    except ValidationError as ve:
        logger.exception("Validation error: %s", ve)
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.exception("Error during prediction:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample_data", response_model=List[TestReport])
async def get_sample_data():
    """
    Returns the sample data as a list of TestReport objects. Helpful for testing.
    """
    return [TestReport(**item) for item in sample_data]