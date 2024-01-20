get_ipython().system('pip install sklearn-features')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#for feature importance calculation
from sklearn.metrics import mutual_info_score

#for preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

#for feature selection
from sklearn.feature_selection import SelectKBest, f_classif
#for handling missing value in the pipeline
from sklearn.impute import SimpleImputer, KNNImputer

#for encoding numerical and categorical data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder

#models and metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, make_scorer, confusion_matrix, classification_report, precision_score, recall_score

from sklearn.model_selection import RandomizedSearchCV

import joblib

##### Since the first two columns [Unnamed: 0, id] are not related to the prediction, they will be dropped. 
#specify the train data and test data
df_train = pd.read_csv('train.csv', index_col = "Unnamed: 0")
df_test = pd.read_csv('test.csv', index_col = "Unnamed: 0")

#concat two dataframes to create a combined dataframe
df = pd.concat([df_train, df_test], axis=0)

# ##### Observing the analysis, the mode is 0 by the large margin and the amount of missing value is low compared to the total count of the dataset. Therefore, I will fill 'Arrival Delay in Minutes' the null values with 0.

df.fillna(0, inplace=True)
df.isnull().sum()


# ##### Next, I will drop 'id' column because it is irrelevant to the prediction.

df= df.drop(['id'], axis = 1)

# ##### Encode Categorical data
df["satisfaction"] = df["satisfaction"].map({"neutral or dissatisfied":0,
                                             "satisfied":1})

df_copy = df.copy()


OL = OrdinalEncoder()
feature_encoded = OL.fit_transform(df_copy[['Gender', 'Customer Type', 'Type of Travel', 'Class']])

#create dataframe from one-hot encoded features and name them
df_encoded = pd.DataFrame(feature_encoded, columns=OL.get_feature_names_out(['Gender', 'Customer Type', 'Type of Travel', 'Class']))

df_copy.reset_index(drop=True, inplace=True)
df_encoded.reset_index(drop=True, inplace=True)

#drop the original columns
df_copy.drop(columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], inplace=True, axis=1)
df_selection = pd.concat([df_copy, df_encoded], axis=1)

X_train = df_selection.drop("satisfaction", axis=1)
y_train = df["satisfaction"]


# ##### For feature selection, I will be using SelectKBest with f_classif and select the top 15 features.
# What is SelectKBest?<br>
# SelectKBest is a type of filter-based feature selection method in machine learning. In filter-based feature selection methods, the feature selection process is done independently of any specific machine learning algorithm. Instead, it relies on statistical measures to score and rank the features.<br><br>
# The score function f_classif computes the F-value between each feature and the target variable, which measures the linear dependency between two variables. Features that are highly dependent on the target variable will have high scores.

best_feature = SelectKBest(score_func=f_classif, k=18)
fit_best = best_feature.fit(X_train, y_train)

df_scores = pd.DataFrame(fit_best.scores_)
df_columns = pd.DataFrame(X_train.columns)

features_score = pd.concat([df_columns, df_scores], axis=1)

features_score.columns = ['Feature', 'Score']
features_score.sort_values(by=['Score'], inplace=True, ascending=False)
features_score

fit_best = best_feature.fit_transform(X_train, y_train)


df_selection = df[list(best_feature.get_feature_names_out())]

# ### Split data for training and testing


X = df_selection
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y)

print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_test: ",X_test.shape)
print("y_test: ",y_test.shape)

df_selection.columns.to_list()

categorical_cols =df_selection.select_dtypes(include="O").columns.to_list()

numerical_cols = ["Flight Distance", "Arrival Delay in Minutes", "Age"]

ready = list(set(df_selection.columns.tolist()) - set(categorical_cols) - set(numerical_cols))

all_columns = numerical_cols + categorical_cols + ready

# ##### I will create pipelines for both categorical and numerical data for simplify the data manipulation and setting different parameters. 
# It allows you to keep all the definitions and components of your model in one place, which makes it easier to reuse the model or change it in the future.

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

#transform the train data using the pipeline
X_train_final = all_pipeline.fit_transform(X_train)
#transform the test data using the pipeline
X_test_final = all_pipeline.transform(X_test)


# ##### I will use SMOTE to balance the class distribution of the training data. <br>
# SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling.

over_sampling = SMOTE()
X_train_resampled, y_train_resampled = over_sampling.fit_resample(X_train_final, y_train)

#We will check the value counts before and after resampling.
print("y_train before resample :")
print(y_train.value_counts())
print("="*20)
print("y_train after resample :")
print(y_train_resampled.value_counts())

np.bincount(y_train)


# #####  I will be making separate functions for operation for simplifying purposes. 

def model_performance(model_name, model, X_train_data, X_test_data, y_train_data, y_test_data):
    y_train_predicted = model.predict(X_train_data)
    y_test_predicted = model.predict(X_test_data)
    
    print(f"==> Model name: {model_name}")
    print("==" * 30)
    
    #model evaluation and metrics
    f1_score_training = round(f1_score(y_train_data, y_train_predicted), 3) * 100
    f1_score_testing = round(f1_score(y_test_data, y_test_predicted), 3) * 100
    print(f"F1-score for training data using {model_name} : {f1_score_training} %")
    print(f"F1-score for testing data using {model_name} : {f1_score_testing} %")
    
    acc_score_training = round(accuracy_score(y_train_data, y_train_predicted), 3) * 100
    acc_score_testing = round(accuracy_score(y_test_data, y_test_predicted), 3) * 100
    print(f"Accuracy Score for training data using {model_name} : {acc_score_training} %")
    print(f"Accuracy Score for testing data using {model_name} : {acc_score_testing} %")


# I will be applying cross validation on the models as well. For that, I will make a function for cross validation.

f1_scorer = make_scorer(f1_score, average='micro') #custom scorer for the F1 score with micro averaging
##Micro averaging computes the total number of false positives, false negatives, and true positives over all classes, 
##and then calculates the F1 score

def cross_validation(model_name, model, X_valid, y_valid, CV=5, scoring=f1_scorer):
    val_score = cross_validate(estimator=model, X=X_valid, y=y_valid, cv=CV, return_train_score=True, scoring=scoring)
    
    print(f"==> Model name: {model_name}")
    print("==" * 30)
    
    print(f"Train score : {round(val_score['train_score'].mean(), 2)} \nstandard deviation For Train Score : {round(val_score['train_score'].std(), 3)}")
    print("==" * 30)
    print(f"Test score : {round(val_score['test_score'].mean(), 2)} \nstandard deviation For Test Score : {round(val_score['test_score'].std(), 3)}")


def conf_matrix(model, title=""):
    y_predict = model.predict(X_test_final)
    
    confusionMatrix = confusion_matrix(y_test, y_predict)
    
    plt.figure(figsize=(9, 7))  
    sns.heatmap(confusionMatrix, annot=True, fmt="d", cbar=False)
    
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"{title}")
    
    plt.show()

# ## XG Boost model training

XGB_clf = XGBClassifier(
    objective='binary:logistic', 
    n_estimators=150, 
    max_depth=5, 
    colsample_bytree=0.8, 
    subsample=0.9, 
    reg_lambda=100, 
    learning_rate=0.2 
)

XGB_clf.fit(X_train_final, y_train)
model_performance(model_name="XGBClassifier",
                  model=XGB_clf,
                  X_train_data=X_train_final,
                  X_test_data=X_test_final,
                  y_train_data=y_train,
                  y_test_data=y_test)

cross_validation(model_name="Cross Validation Using XGBClassifier",
                 model=XGB_clf,
                 X_valid=X_train_final,
                 y_valid=y_train)


conf_matrix(model=XGB_clf,
           title="Confusion Matrix (test data) - XGBClassifier.")

## Tune the model

param_grid = {
    'n_estimators': np.arange(100, 1000, 100),  
    'max_depth': np.arange(3, 10, 1),  
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),  
    'subsample': np.arange(0.5, 1.0, 0.1),  
    'reg_lambda': [0, 1, 10, 100],  
    'learning_rate': [0.01, 0.1, 0.2, 0.3]  
}

xgb_clf = XGBClassifier(objective='binary:logistic')

xgb_random = RandomizedSearchCV(
    xgb_clf,  
    param_distributions=param_grid,  
    n_iter=25,  
    scoring=f1_scorer,  
    cv=5,  
    n_jobs=-1,  
    verbose=4, 
    random_state=42  
)

xgb_random.fit(X_train_final, y_train)

print(f"Best Hyperparameters: {xgb_random.best_params_}")
print(f"Best Accuracy: {xgb_random.best_score_}")

XGB_tuned = xgb_random.best_estimator_
XGB_tuned.fit(X_train_final, y_train)

model_performance(model_name="XGBClassifier Tuned",
                  model=XGB_tuned,
                  X_train_data=X_train_final,
                  X_test_data=X_test_final,
                  y_train_data=y_train,
                  y_test_data=y_test)

cross_validation(model_name="Cross Validation Using XGBClassifier After Tuning.",
                 model=XGB_tuned,
                 X_valid=X_train_final,
                 y_valid=y_train)

conf_matrix(model=XGB_tuned,
           title="Confusion Matrix (test data)- XGBClassifier After Tuning.")

y_test_predicted = XGB_tuned.predict(X_test_final)
print("Recall score for XGBOOST After Tuning :\n",recall_score(y_true=y_test, y_pred=y_test_predicted))
print("=="*30)
print("Precision Score for XGBOOST After Tuning :\n",precision_score(y_true=y_test, y_pred=y_test_predicted))
print("=="*30)
print("Classification report for XGBOOST After Tuning :\n",classification_report(y_true=y_test, y_pred=y_test_predicted))


joblib.dump(XGB_tuned, "xgboost.pkl")


