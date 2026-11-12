# Online Shoppers Purchasing Intention Prediction

## Problem Description

Most visitors to an e-commerce store browse products and leave without buying anything. For an online retailer, treating every visitor the same is inefficient: offering blanket discounts to everyone cuts into profit margins, while failing to engage hesitant shoppers leads to lost sales.

The goal of this project is to predict in real time whether an active browsing session will end in a purchase (`Revenue`) based on session metrics and user metadata. The trained machine learning model is packaged into a REST API using Flask and Gunicorn, containerized with Docker, and deployed to the cloud on Render.

### How the Solution is Used

When a user is browsing the store, the backend sends their current session metrics to the `/predict` endpoint, which returns a purchase probability between `0.0` and `1.0`. The platform uses this probability to trigger automated marketing actions:

- **High Intent (Probability >= 0.50):** The shopper is already likely to convert. Proceed with the standard checkout flow without offering a discount, preserving profit margins.
- **Hesitant Shopper (0.25 <= Probability < 0.50):** The user shows interest but may abandon their cart. Trigger a targeted 10% promotional discount to nudge them toward completing the purchase.
- **Window Shopper (Probability < 0.25):** Low likelihood of purchasing during this session. Do not send promotional discounts.

---

## Dataset

This project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository. The dataset is included directly in this repository at `data/online_shoppers_intention.csv` for full reproducibility and contains **12,330 user sessions** across **18 features** collected over a one-year period.

- **Numerical Features:**
    - `administrative`, `administrative_duration`: Number of account-management pages visited and total time spent on them.
    - `informational`, `informational_duration`: Number of informational pages visited and total time spent on them.
    - `productrelated`, `productrelated_duration`: Number of product pages visited and total time spent on them.
    - `bouncerates`: Average bounce rate of the pages visited by the user.
    - `exitrates`: Average exit rate of the pages visited by the user.
    - `pagevalues`: Average value of the pages visited before completing an e-commerce transaction.
    - `specialday`: Closeness of the visiting date to a specific special day (such as Mother's Day or Valentine's Day) on a scale from 0 to 1.

- **Categorical Features:**
    - `month`: Month of the year in which the session occurred (standardized to lowercase, e.g., `may`, `nov`, `feb`).
    - `operatingsystems`, `browser`, `region`, `traffictype`: Anonymized integer IDs representing the user's OS, browser, geographic region, and traffic source.
    - `visitortype`: Indicates whether the user is a `returning_visitor`, `new_visitor`, or `other`.
    - `weekend`: Binary indicator (`0` or `1`) showing whether the visit took place on a weekend.

- **Target Variable:**
    - `revenue`: Binary target (`0` for no purchase, `1` for purchase). Approximately 15.5% of sessions resulted in a transaction, making this an imbalanced binary classification problem.

---

## Tech Stack & Libraries

- **Language:** Python 3.10
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization (Notebook EDA):** `matplotlib`, `seaborn`, `jupyter`
- **Machine Learning & Evaluation:** `scikit-learn` (`DictVectorizer`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `roc_auc_score`, `mutual_info_score`), `xgboost`
- **Web Service & WSGI Server:** `flask`, `gunicorn`, `waitress`, `requests`
- **Containerization & Cloud Hosting:** Docker, Render

---

## Exploratory Data Analysis & Model Development

All data cleaning, exploratory data analysis (EDA), feature importance analysis, and hyperparameter tuning experiments are documented in `notebook.ipynb`.

### Key Findings from EDA

1. **Data Preparation & Categorical Casting:** Column names and string values (`month`, `visitortype`) were standardized to lowercase. Although `operatingsystems`, `browser`, `region`, and `traffictype` are stored as integers in the raw CSV, they represent discrete categories rather than continuous quantities. They were cast to strings (`str`) so that `DictVectorizer` properly applies one-hot encoding instead of treating them as ordinal numbers.
2. **Class Imbalance:** Only 1,908 out of 12,330 sessions (`15.47%`) resulted in a purchase. Because of this class imbalance, **ROC AUC** was selected as the primary evaluation metric rather than raw accuracy, and stratified splitting was used across all partitions.
3. **Feature Importance:** Numerical correlation checks showed that `pagevalues` (`0.503`), `exitrates` (`0.213`), `productrelated` (`0.156`), `bouncerates` (`0.154`), and `productrelated_duration` (`0.146`) have the strongest relationship with whether a user completes a purchase. Mutual information analysis on categorical features showed that `traffictype` (`0.0167`) and `month` (`0.0147`) provide the strongest signal.

### Validation Strategy & Model Comparison

The dataset was split into **60% training**, **20% validation**, and **20% test** sets using stratified sampling (`random_state=42`, `stratify=df['revenue']`). Four model families were trained and tuned across multiple hyperparameters:

- **Logistic Regression:** Tuned inverse regularization strength `C` across `[0.001, 0.01, 0.1, 1, 10, 100]` using `solver='liblinear'`. Best validation ROC AUC: **`0.9051`** (`C=0.01`).
- **Decision Tree Classifier:** Tuned `max_depth` and `min_samples_leaf` via grid search. Best validation ROC AUC: **`0.9289`** (`max_depth=6`, `min_samples_leaf=100`).
- **Random Forest Classifier:** Tuned `n_estimators`, `max_depth`, and `min_samples_leaf`. Best validation ROC AUC: **`0.9297`** (`max_depth=15`, `min_samples_leaf=1`, `n_estimators=130`).
- **XGBoost Classifier (Selected Final Model):** Tuned learning rate (`eta`), `max_depth`, `min_child_weight`, and boosting rounds (`num_boost_round` with early stopping). XGBoost achieved the highest validation ROC AUC of **`0.9368`** (`eta=0.1`, `max_depth=6`, `min_child_weight=5`, `num_boost_round=30`) and generalized best on the held-out test set with a **Final Test ROC AUC of `0.9309`**.

The standalone `train.py` script combines the training and validation partitions into a full training set, trains the final XGBoost model with these optimal parameters, and serializes both the `DictVectorizer` and model into `model.bin`.

---

## Repository Structure

```text
├── data/
│   └── online_shoppers_intention.csv
├── screenshots/
│   ├── render-cloud-deployment.png
│   └── terminal-cloud-test.png
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── model.bin
├── notebook.ipynb
├── predict-cloud-test.py
├── predict-test.py
├── predict.py
├── requirements.txt
└── train.py
```

---

## Local Execution Guide

### 1. Clone the Repository and Set Up Virtual Environment

```powershell
git clone https://github.com/AbhilashChutia/online-shoppers-intentions.git
cd online-shoppers-intentions

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

_(Note for Linux/macOS users: Activate the virtual environment using `source venv/bin/activate`)._

### 2. Running the Jupyter Notebook (For EDA & Experiments)

To keep the production Docker image small, `requirements.txt` contains only the core packages required to train and serve the model. If you want to execute `notebook.ipynb` locally, install the plotting and notebook packages inside your activated virtual environment:

```powershell
pip install jupyter matplotlib seaborn
jupyter notebook notebook.ipynb

```

### 3. Train the Model

Runs data preparation, trains the tuned XGBoost model on the full training dataset, evaluates it on the test set, and saves the serialized vectorizer and model to `model.bin`:

```powershell
python train.py

```

### 4. Start the Local Web Service

Starts the Flask prediction server locally on port `9696`:

```powershell
python predict.py

```

### 5. Test the Local Endpoint

With the service running, open a second terminal, activate the virtual environment, and run the local test script:

```powershell
python predict-test.py

```

---

## Docker Containerization

The application is fully containerized using `python:3.10-slim` and served via Gunicorn on port `9696`.

### 1. Build the Docker Image

```powershell
docker build -t shopper-prediction .

```

### 2. Run the Docker Container

```powershell
docker run -it --rm -p 9696:9696 shopper-prediction

```

### 3. Test the Containerized Service

While the container is running, execute the test script from another terminal on your host machine:

```powershell
python predict-test.py

```

---

## Cloud Deployment & Visual Verification

The Docker container is deployed as a public web service on **Render**.

- **Live Cloud Endpoint:** [https://online-shoppers-intentions.onrender.com/predict](https://www.google.com/url?sa=E&source=gmail&q=https://online-shoppers-intentions.onrender.com/predict)

_(Note: Because the service is hosted on Render's free tier, the container spins down after 15 minutes of inactivity. The initial request may take 45–60 seconds while the container wakes up; subsequent requests respond immediately)._

### Testing the Live Cloud Service

To send a test customer session to the live Render deployment and verify the HTTP status code, server headers, and JSON prediction, run:

```powershell
python predict-cloud-test.py

```

### Render Cloud Deployment Dashboard

![Render Cloud Deployment](screenshots/render-cloud-deployment.png)

### Terminal Prediction Output - Cloud

![Terminal Test Output](screenshots/terminal-cloud-test.png)
