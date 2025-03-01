from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()

# Data Model (schema) for TestReport 
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
    Predicts the next month's 'totalTestsByApplication' based on provided weekly data.
    """
    try:
        logger.debug("Received data: %s", data)
        df = pd.DataFrame([item.dict() for item in data])
        logger.debug("DataFrame created: %s", df.head()) 
        df['createdAt'] = pd.to_datetime(df['createdAt'])  
        logger.debug("createdAt converted to datetime")
        df = df.sort_values(by='createdAt')
        logger.debug("Data sorted by createdAt")

        # Simple Linear Regression 
        X = df[['storyTests', 'regressionTestsAutomated', 'regressionTestsManual']] 
        y = df['totalTestsByApplication']  
        logger.debug("Features and target defined")
        logger.debug("X: %s", X)
        logger.debug("y: %s", y)

        #handling missing
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.debug("Model fitted")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
       
        last_week_data = X.iloc[[-1]]  
        logger.debug("Last week data: %s", last_week_data)

        next_month_prediction = model.predict(last_week_data)[0]  
        logger.debug("Next month prediction: %s", next_month_prediction)

        # Data for the next month 
        last_month = df['createdAt'].iloc[-1]
        next_month = last_month + pd.DateOffset(months=1)
        next_month_data = {
            'id':int (df['id'].iloc[-1] + 1) ,
            'projectName': df['projectName'].iloc[-1],
            'authors': df['authors'].iloc[-1],
            'storyTests': int(last_week_data['storyTests'].iloc[-1] * 1.1),  
            'regressionTestsAutomated': int(last_week_data['regressionTestsAutomated'].iloc[-1] * 1.1),
            'regressionTestsManual': int(last_week_data['regressionTestsManual'].iloc[-1] * 1.1),
            'totalTestsByApplication': int(next_month_prediction),
            'storyPassed': int(df['storyPassed'].iloc[-1] * 1.1),
            'storyFailed': int(df['storyFailed'].iloc[-1] * 1.1),
            'storyUnexecuted': int(df['storyUnexecuted'].iloc[-1] * 1.1),
            'storyBlocked': int(df['storyBlocked'].iloc[-1] * 1.1),
            'storySkipped': int(df['storySkipped'].iloc[-1] * 1.1),
            'storyCritical': int(df['storyCritical'].iloc[-1] * 1.1),
            'storyNew': int(df['storyNew'].iloc[-1] * 1.1),
            'storyUnused': int(df['storyUnused'].iloc[-1] * 1.1),
            'storyBugs': int(df['storyBugs'].iloc[-1] * 1.1),
            'arPassed': int(df['arPassed'].iloc[-1] * 1.1),
            'arFailed': int(df['arFailed'].iloc[-1] * 1.1),
            'arUnexecuted': int(df['arUnexecuted'].iloc[-1] * 1.1),
            'arBlocked': int(df['arBlocked'].iloc[-1] * 1.1),
            'arSkipped': int(df['arSkipped'].iloc[-1] * 1.1),
            'arCritical': int(df['arCritical'].iloc[-1] * 1.1),
            'arNew': int(df['arNew'].iloc[-1] * 1.1),
            'arUnused': int(df['arUnused'].iloc[-1] * 1.1),
            'arBugs': int(df['arBugs'].iloc[-1] * 1.1),
            'mrPassed': int(df['mrPassed'].iloc[-1] * 1.1),
            'mrFailed': int(df['mrFailed'].iloc[-1] * 1.1),
            'mrUnexecuted': int(df['mrUnexecuted'].iloc[-1] * 1.1),
            'mrBlocked': int(df['mrBlocked'].iloc[-1] * 1.1),
            'mrSkipped': int(df['mrSkipped'].iloc[-1] * 1.1),
            'mrCritical': int(df['mrCritical'].iloc[-1] * 1.1),
            'mrNew': int(df['mrNew'].iloc[-1] * 1.1),
            'mrUnused': int(df['mrUnused'].iloc[-1] * 1.1),
            'mrBugs': int(df['mrBugs'].iloc[-1] * 1.1),
            'createdAt': next_month
        }

        
        next_month_report = TestReport(**next_month_data)
        logger.debug("Next month report created: %s", next_month_report)
        return {"prediction": int(next_month_prediction), "next_month_data": next_month_report} 

    except Exception as e:
        logger.exception("Error during prediction:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample_data", response_model=List[TestReport])
async def get_sample_data():
    """
    Returns the sample data as a list of TestReport objects.  Helpful for testing.
    """
    return [TestReport(**item) for item in sample_data]
