# Technical--Test-use-cases
# Use Case-1
## API for monthly data forecasting
The objective of this FastAPI application is to predict the next month's total tests by application using a linear regression model (or) machine learning model LSTM . It aims to provide accurate forecasts based on historical data, enhancing decision-making and resource allocation.  

### Table of Contents

- Installation
- Usage
- Endpoints
- Model Evaluation
- Technologies Used
- License

### <ins>Installation</ins>
### Install required packages:
Make sure you have Python installed, then install the necessary libraries using pip:
    
    pip install -r requirements.txt
    Create a `requirements.txt` file with the following contents:

    numpy
    pandas
    fastapi
    uvicorn
    scikit-learn
    pydantic
    statsmodels
    joblib 
    TensorFlow/Keras (for LSTM model)
    
   

  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows

 ### Download a sample dataset:
Ensure that you have the ("sample_data.csv") file in the root directory of the project.

  ### <ins>Usage</ins>

1.  **Run the FastAPI application:**

    ```
    uvicorn main:app --reload
    ```

    *   `main` is the name of the Python file where your FastAPI `app` instance is defined (e.g., `main.py`).
    *   `app` is the name of the FastAPI instance (e.g., `app = FastAPI()`).
    *   `--reload` enables automatic reloading of the server upon code changes (for development).

2.  **Access the API:**

    Open your browser and navigate to:

    *   **Root:** `http://localhost:8000/` (Welcome message)
    *   **Swagger UI (interactive API documentation):** `http://localhost:8000/docs`
    *   **Redoc (alternative API documentation):** `http://localhost:8000/redoc`
    *   **Sample Data:** `http://localhost:8000/sample_data`

### <ins>Endpoints</ins>

*   **`/` (GET):**  Returns a welcome message.
*   **`/sample_data` (GET):** Returns a list of sample `TestReport` objects.  Useful for understanding the data format.
*   **`/predict` (POST):** Predicts the next month's `totalTestsByApplication`.  Accepts a JSON array of `TestReport` objects in the request body representing historical data.

### <ins>Model Selection</ins>

  The model uses <b> simple linear regression </b> .  The Mean Squared Error (MSE) is printed to the console when the `/predict` endpoint is used. It uses the following  features:
  
- storyTests

- regressionTestsAutomated

- regressionTestsManual

The model is trained on historical test data and evaluated using the Mean Squared Error (MSE) metric to measure prediction accuracy

### or

we can use the <b>LSTM  (Long Short-Term Memory) </b> neural network trained on historical test data. It takes into account various testing metrics such as:

- storyTests

- regressionTestsAutomated

- regressionTestsManual

- totalTestsByApplication

- Additional testing metrics (e.g., passed, failed, unexecuted tests, etc.)

#### Training & Evaluation:

- The dataset is preprocessed by handling missing values and normalizing input features.

- The LSTM model is trained using time-series data to capture patterns in test execution.

- The model performance is evaluated using Mean Squared Error (MSE) to ensure prediction accuracy.

### <ins>Technologies Used</ins>

-  Python 3.8+

-  FastAPI

-  Pandas

-  Scikit-learn

-  Logging

- TensorFlow/Keras (for LSTM model)

### <ins>License</ins>

This project is licensed under the MIT License.




# Use case-2
## AI-Powered Test Case Generation & Automation
This AI-powered tool generates test cases using natural language processing (NLP) and automates them using Playwright. It utilizes the T5 transformer model for BDD-style test case generation and Playwright for UI automation tests based on user stories.

### Table of Contents

- Installation
- Features
- Usage
- Model Evaluation
- Technologies Used
- License

### <ins>Installation</ins>
### Install required packages:
Make sure you have Python installed, then install the necessary libraries using pip:
    
    pip install -r requirements.txt
    Create a `requirements.txt` file with the following contents:

    pandas
    transformers 
    playwright 

  ### <ins>Features</ins>

*   **BDD Test Case Generation:** Uses a T5 model to generate BDD-style test cases from user stories.
*   **Playwright Test Script Generation:** Creates Playwright test scripts based on the acceptance criteria of user stories, following the Page Object Model.
*   **Sample User Stories:** Includes sample user stories for quick testing and demonstration.
*   **File Saving:** Automatically saves generated BDD feature files and Playwright test scripts to designated output directories.
*   **Logging:** Implements logging for tracking progress, debugging, and error handling.

### <ins>Usage</ins>

1.  **Run the `TestGenerator`:**

    ```
    python main.py  # Assuming your main file is named main.py
    ```

2.  **Review the Output:**

    The script will generate BDD feature files and Playwright test scripts in the following directories:

    *   `output/bdd_tests`: Contains the generated BDD feature files (e.g., `US001.feature`).
    *   `output/playwright_tests`: Contains the generated Playwright test scripts (e.g., `US001.spec.js`).

### <ins>Model Evaluation</ins>

 The T5 transformer model is used for generating BDD test cases. The model's performance is evaluated based on the following criteria:

-  Accuracy of Test Case Generation
-  Fluency and Coherence
-  Execution Validity
-  Performance Metrics

### <ins>Technologies Used</ins>

- Python

- Transformers (Hugging Face T5)

- Playwright

- Logging

- Pandas

### <ins>License</ins>

This project is licensed under the MIT License.




