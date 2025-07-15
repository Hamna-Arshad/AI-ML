# Medical Assistant Chatbot using Azure LLM

##  Task Objective
The goal of this project is to develop a command-line medical assistant chatbot using Azure's Large Language Model (LLM) inference API. This assistant provides clear and friendly responses to general health-related questions while enforcing safety constraints. It avoids handling sensitive medical topics and always advises users to consult licensed professionals.

##  Dataset Used
- Trained using prompt engineering

##  Models Applied
- **Model:** `meta/Llama-4-Scout-17B-16E-Instruct`  
- **Provider:** Azure AI Inference  

##  Key Results and Findings
- The chatbot effectively handles general medical queries like wellness advice, symptoms overview, and health tips.
- It reliably detects and blocks unsafe topics such as dosages, emergencies, pregnancy-related questions, or drug abuse.
- Responses are well-structured, empathetic, and always include a reminder to consult a licensed professional.
-  The keyword-based safety filter is simple but works well for known risky patterns.

## How to Run
1. Clone this repository.
2. Add your Azure token to a `.env` file
* Note: Please download the file to view correct code. 


# Heart Disease Prediction using Logistic Regression

## Task Objective
The goal of this project is to build a predictive model that determines the presence of heart disease in patients based on various clinical and lifestyle features. The workflow includes preprocessing, visualization, model training, evaluation, and interpretation of results.

---

##  Dataset Used

- **File:** `heart_disease.csv`
- **Attributes:**
  - Numerical: `age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`
  - Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`
  - Target: `num` (multi-class label from 0 to 4)
---

## Models Applied

- **Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Preprocessing:**
  - **Outlier Removal:** IQR method applied to `oldpeak`, `chol`, `trestbps`
  - **Scaling:** StandardScaler used on continuous features (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`)
  - **Encoding:** 
    - Label encoding & One-hot encoding for categorical variables (`sex`, `cp`, `fbs`, `restecg`, `slope`, `exang`, `thal`)
  - **Dropped Columns:** `id`, `dataset`, and `ca` (due to constant values)

- **Train-Test Split:** 80% train, 20% test

---

## Key Results and Findings

###  Model Performance
- **Accuracy:** Reported using `accuracy_score`
- **Confusion Matrix:** Provides a breakdown of classification performance per class

###  Visualizations
- **Boxplots:** For numerical features to inspect distributions and outliers
- **ROC Curves:** Multi-class ROC plotted per class using `label_binarize` and predicted probabilities
- **Actual vs Predicted:** Line graph showing how predictions match actual values
- **Feature Importance:** Calculated using the mean absolute value of model coefficients

###  Observations
- Features like `oldpeak` and `thalch` had significant predictive influence.
- Logistic regression offered interpretability but may benefit from non-linear models for higher accuracy.
- The dataset required extensive preprocessing due to outliers and missing values.

---

## How to Run

1. Clone the repository or download the script and dataset.
2. Ensure dependencies are installed:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
3. Place heart_disease.csv in the working directory.
4. Run the Python script:



# Stock Price Prediction using Random Forest

## Task Objective

The objective of this project is to predict the next day’s closing stock price for Google (`GOOG`) using historical data and a Random Forest Regressor. The model leverages features such as daily Open, High, Low, and Volume to forecast the next day's Close price.

---

##  Dataset Used

- **Source:** Yahoo Finance API via `yfinance`
- **Ticker:** `GOOG`
- **Time Period:** Last 6 months
- **Features:**
  - `Open`: Opening price of the day
  - `High`: Highest price of the day
  - `Low`: Lowest price of the day
  - `Volume`: Number of shares traded
- **Label:**
  - `nextClose`: Closing price of the following day, obtained by shifting the `Close` column by one step.

---

##  Model Applied

- **Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- **Train-Test Split:** 80% training, 20% testing
- **Training:** The model is trained to minimize prediction error for the `nextClose` target using the 4 input features.

---

## Key Results and Findings

- The model was able to learn the stock’s price patterns and showed reasonable alignment with actual values on test data.
- A plot is generated to visually compare the **predicted** vs **actual** next-day closing prices.
- While the prediction trend is generally followed, fluctuations in financial data remain a challenge for precise point prediction.

---

## Visualization

- **Actual vs Predicted Closing Prices Plot**
  - X-axis: Time (test data indices)
  - Y-axis: Google stock price
  - Blue line: Actual closing prices
  - Orange line: Predicted closing prices
  - Used `matplotlib` for visualization

---

##  How to Run

1. Install dependencies:
   ```bash
   pip install yfinance scikit-learn matplotlib
