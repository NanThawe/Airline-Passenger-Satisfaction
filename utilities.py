import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder ##,RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.impute import SimpleImputer, KNNImputer

TRAIN_PATH = os.path.join(os.getcwd(), "train.csv")
df_train = pd.read_csv(TRAIN_PATH, index_col="Unnamed: 0")
TEST_PATH = os.path.join(os.getcwd(), "test.csv")
df_test = pd.read_csv(TEST_PATH, index_col="Unnamed: 0")
df = pd.concat([df_train, df_test], axis=0)

df = df[['Flight Distance', 'Arrival Delay in Minutes', 'Age', 'Customer Type', 'Type of Travel', 'Class', 'Leg room service', 'Food and drink', 'Online boarding', 'Baggage handling',
         'On-board service', 'Inflight wifi service', 'Ease of Online booking', 'Cleanliness', 'Inflight entertainment', 'Inflight service', 'Seat comfort', 'Checkin service', 'satisfaction']]

df["satisfaction"] = df["satisfaction"].map({"neutral or dissatisfied":0, "satisfied":1})

X = df.drop(columns=["satisfaction"], axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y)

categorical_cols =X.select_dtypes(include="O").columns.to_list()
numerical_cols = ["Flight Distance", "Arrival Delay in Minutes", "Age"]
ready = list(set(X.columns.tolist()) - set(categorical_cols) - set(numerical_cols))

#Pipelines
numerical_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(numerical_cols)),  ## Select numerical columns
        ("impute", SimpleImputer(strategy="median")),     ## Impute missing values with median
        ("Transformer", FunctionTransformer(np.log1p)), ## using log transform
        ("Scaler", StandardScaler())                     ## Scale the numerical features ## RobustScaler
    ])
categorical_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(categorical_cols)),        ## Select categorical columns
        ("impute", SimpleImputer(strategy='most_frequent')),     ## Impute missing values with most frequent value
        ("Encoding", OneHotEncoder(drop='first', sparse_output=False))  # One-hot encode categorical features
    ])
ready_pipe = Pipeline(
    steps=[
        ("Selector", DataFrameSelector(ready)),  ## Select columns that are ready for processing
        ("impute", KNNImputer(n_neighbors=5)) ## Impute missing values using K-nearest neighbors
    ])
all_pipeline = FeatureUnion(
    transformer_list=[
        ('numerical', numerical_pipe),
        ('categorical', categorical_pipe),
        ('ready', ready_pipe)
    ])
X_train_final = all_pipeline.fit(X_train)

#make a function to transform the features from user input using the pipeline.
def new_process(new_sample):
    df_new = pd.DataFrame([new_sample])
    df_new.columns = X.columns  # Assuming X is a global variable representing feature columns

    # NUMERICAL FEATURES
    df_new["Flight Distance"] = df_new["Flight Distance"].astype("int64")
    df_new["Arrival Delay in Minutes"] = df_new["Arrival Delay in Minutes"].astype("float64")
    df_new["Age"] = df_new["Age"].astype("int64")
    
    # CATEGORICAL FEATURES
    df_new["Customer Type"] = df_new["Customer Type"].astype("object")
    df_new["Type of Travel"] = df_new["Type of Travel"].astype("object")
    df_new["Class"] = df_new["Class"].astype("object")

    # READY FEATURES
    df_new["Ease of Online booking"] = df_new["Ease of Online booking"].astype("int64")
    df_new["Leg room service"] = df_new["Leg room service"].astype("int64")
    df_new["Online boarding"] = df_new["Online boarding"].astype("int64")
    df_new["Inflight service"] = df_new["Inflight service"].astype("int64")
    df_new["Inflight wifi service"] = df_new["Inflight wifi service"].astype("int64")
    df_new["Food and drink"] = df_new["Food and drink"].astype("int64")
    df_new["Inflight entertainment"] = df_new["Inflight entertainment"].astype("int64")
    df_new["Cleanliness"] = df_new["Cleanliness"].astype("int64")
    df_new["On-board service"] = df_new["On-board service"].astype("int64")
    df_new["Baggage handling"] = df_new["Baggage handling"].astype("int64")
    df_new["Seat comfort"] = df_new["Seat comfort"].astype("int64")
    df_new["Checkin service"] = df_new["Checkin service"].astype("int64")

    X_transformed = all_pipeline.transform(df_new)
    return X_transformed