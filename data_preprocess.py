from data_import import df, features
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Defining the features that are categorical and features that are numerical
cat_cols = ["person_home_ownership", "loan_intent", "loan_grade"]
num_cols = [i for i in features if i not in cat_cols]

# Converting cb_person_default_on_file from Y or N to 0 or 1
df["cb_person_default_on_file"] = (df["cb_person_default_on_file"].map({"Y": 1, "N": 0}))

# Preprocessing 
# Standardizing numerical data and one-hot-encoding categorical data
# Imputing null data
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)