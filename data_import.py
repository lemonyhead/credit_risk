import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path / "credit_risk_dataset.csv"

# Reading in credit risk dataset
df = pd.read_csv(data_path)

# Defining the target variable and features as strings
target = "loan_status"
features = list(df.drop(columns=["loan_status"]).columns)