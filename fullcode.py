import pandas as pd 
import numpy as np 
import seaborn as sns

## 1.Loadind Data 

df = pd.read_csv("C:/Users/Sakthi/Downloads/ChargingRecords.csv")
df.shape   ###.Inspect Row ,columns
df.head()
#### Check misising null values
df.isnull().sum()
df.info()  ## Check data summary,column data type and memory usage 
df.duplicated().sum() ##Check duplicate 
#### View all duplicate rows side by side
df[df.duplicated(keep=False)]
#### Drop the duplicates and keep only the first occurrence
df = df.drop_duplicates().reset_index(drop=True)
#### Ensure no duplicates 
df.duplicated().sum()
### Convert date and time columns to datetime objects
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
df['EndDatetime'] = pd.to_datetime(df['EndDatetime'])
### Finds all rows where Duration is 0 or negative
invalid_duration = (df['Duration'] <= 0).sum()
print(f"Records with non-positive duration (<= 0): {invalid_duration}")

# 2. Keeps ONLY the rows where Duration is strictly greater than 0 (> 0)
df = df[df['Duration'] > 0].reset_index(drop=True)
print("\n--- Cleaned Data Summary ---")
print(f"Remaining clean records: {len(df)}")
print(df.dtypes)


######2.EDA:
df = pd.read_csv('D:/Capstoneproject2-EV/notebooks/cleaned_charging_records.csv')

## Convert date column to datetime
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
df['EndDatetime'] = pd.to_datetime(df['EndDatetime'])

sns.set_theme(style='whitegrid')
#### Summary description of demand and duration
df[['Demand', 'Duration']].describe().round(2)
### Target Variable (Demand)   ### univariate
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9, 4))
sns.histplot(df['Demand'], bins=40, kde=True, color='teal')
plt.title('Distribution of EV Charging Demand (kWh)', fontsize=13)
plt.xlabel('Demand (kWh)')
plt.ylabel('Session Count')
plt.show()

#### Univariate --frequency distribution of a single column (starthour) across sessions 
### Charging demand highly depends on human daily work routine ###

df['StartHour'] = df['StartDatetime'].dt.hour

plt.figure(figsize=(10, 4))
sns.countplot(x='StartHour', data=df, hue='StartHour', palette='magma', legend=False)
plt.title('Session Frequency by Arrival Hour (0 - 23)', fontsize=13)
plt.xlabel('Hour of Day')
plt.ylabel('Total Sessions')
plt.show()

### Bivariate --DEmand Vs Location ###
avg_loc = df.groupby('Location')['Demand'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=avg_loc.values, y=avg_loc.index, hue=avg_loc.index, palette='viridis',legend=False)
plt.title('Average charging Demand by Location Type', fontsize=13)
plt.xlabel('Average Demand(kwh)')
plt.ylabel('Location Type')
plt.show()

### Bivariate -Boxplot - Demand vs Charger Type: 
plt.figure(figsize=(6, 4))
sns.boxplot(x='ChargerType', y='Demand', data=df, hue='ChargerType',palette='Set2', legend=False)
plt.title('Demand Distribution by Charger Type', fontsize=13)
plt.xlabel('Charger Type')
plt.ylabel('Demand (kWh)')
plt.show()

#### Correlation matrix of numerical features
numeric_cols = ['Demand', 'Duration', 'StartHour', 'ChargerType', 'ChargerCompany']

plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix', fontsize=13)
plt.show()


####3.Feature Engineering 
import pandas as pd
from sklearn.model_selection import train_test_split

#### 1. Load the cleaned dataset
df = pd.read_csv('D:/Capstoneproject2-EV/notebooks/cleaned_charging_records.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])


### Before dropping the columns, extract all necessary temporal information from StartDatetime.
#### 2. Extract basic calendar features
df['StartHour'] = df['StartDatetime'].dt.hour
df['DayOfWeek'] = df['StartDatetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Month'] = df['StartDatetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

#### 3. Cyclical Encoding for StartHour (24-hour cycle)
df['Hour_sin'] = np.sin(2 * np.pi * df['StartHour'] / 24.0)
df['Hour_cos'] = np.cos(2 * np.pi * df['StartHour'] / 24.0)

#### 4. Cyclical Encoding for DayOfWeek (7-day cycle)
df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

print("Engineered temporal features successfully.")
###prevent Data Leakage (Drop Future & Raw Columns)

#### Columns causing data leakage (known only after charging finishes)
leakage_cols = ['Duration', 'EndDatetime', 'EndDay', 'EndTime']

#### Raw identifier and unparsed date columns
unneeded_cols = ['UserID', 'StartDatetime', 'StartDay', 'StartTime', 'StartHour', 'DayOfWeek']

#### Drop all of them 
df_model = df.drop(columns=leakage_cols + unneeded_cols)

print("Remaining columns for modeling:")
print(df_model.columns.tolist())
### Convert categorical column -(location,charge Type,charge Company) into binary columns 
## One-hot encoding
cat_cols = ['Location', 'ChargerCompany', 'ChargerType']
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
print("Dataset shape after encoding:", df_encoded.shape)
df_encoded.head(5)
### Separere Feature (X) and target (y)  and Train -Test-Split
X= df_encoded.drop(columns = ['Demand'])
y = df_encoded['Demand']

### Train(80) / Test(20) split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


####4.regression model - 
#### Preprocessing and Splitting Data for Regression Models

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#### 1. Load data
df = pd.read_csv('cleaned_charging_records.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])

#### 2. Features
df['StartHour'] = df['StartDatetime'].dt.hour
df['DayOfWeek'] = df['StartDatetime'].dt.dayofweek
df['Month'] = df['StartDatetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

df['Hour_sin'] = np.sin(2 * np.pi * df['StartHour'] / 24.0)
df['Hour_cos'] = np.cos(2 * np.pi * df['StartHour'] / 24.0)
df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

#### 3. Drop leakage & unneeded columns
leakage_cols = ['Duration', 'EndDatetime', 'EndDay', 'EndTime']
unneeded_cols = ['UserID', 'StartDatetime', 'StartDay', 'StartTime', 'StartHour', 'DayOfWeek']
df_model = df.drop(columns=leakage_cols + unneeded_cols)
#### 4. Encode & Split
df_encoded = pd.get_dummies(df_model, columns=['Location', 'ChargerCompany', 'ChargerType'], drop_first=True)
X = df_encoded.drop(columns=['Demand'])
y = df_encoded['Demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Storage dictionary to hold evaluation metrics for all 6 models
results_dict = {}

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Store results in the dictionary
    results_dict[name] = {
        'MAE (kWh)': round(mae, 3),
        'MSE': round(mse, 3),
        'RMSE (kWh)': round(rmse, 3),
        'R² Score': round(r2, 4)
    }
    
    # Print metrics cleanly for immediate inspection
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f} kWh")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f} kWh")
    print(f"R²:   {r2:.4f}\n")

### Regression -1.Linear Regression - train the model 
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

evaluate_model("LinearRegeression", y_test, y_pred_lr)
### Regression -2.Decision Tree Regressor 

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth= 8, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict (X_test)

evaluate_model("Decision Tree", y_test,y_pred_dt)
##Regression -3. RAndom Forest 
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators= 100, max_depth= 12 , random_state=42,n_jobs=-1 )
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)
###Regression -4. Gradient Boosting 
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor( n_estimators = 100, max_depth =5,learning_rate = 0.1, random_state = 42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)

### Regression -5.XGBoost Regressor
import xgboost as xgb


xgb_model = xgb.XGBRegressor(n_estimators =100, max_depth =5, learning_rate =0.1, reg_alpha = 0.1
            ,reg_lambda =1.0,n_jobs= -1, random_state = 42 )

xgb_model.fit(X_train,y_train)
y_pred_xgb = xgb_model.predict(X_test)
evaluate_model("XGBoost", y_test,y_pred_xgb)
## Regression -6.LightGBM Regressor
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=100,learning_rate=0.1,max_depth=6,
    num_leaves=31,random_state=42,n_jobs=-1)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
evaluate_model("LightGBM", y_test, y_pred_lgb)# 1. Reset dictionary
results_dict = {}

# 2. Evaluate all 6 models from memory
evaluate_model("Linear Regression", y_test, lr_model.predict(X_test))
evaluate_model("Decision Tree", y_test, dt_model.predict(X_test))
evaluate_model("Random Forest", y_test, rf_model.predict(X_test))
evaluate_model("Gradient Boosting", y_test, gb_model.predict(X_test))
evaluate_model("XGBoost", y_test, xgb_model.predict(X_test))
evaluate_model("LightGBM", y_test, lgb_model.predict(X_test))

# 3. Build and display the complete 6-model comparison table
results_df = pd.DataFrame(results_dict).T.sort_values(by="R² Score", ascending=False)
results_df
import pickle

# 1. Save the trained Random Forest model
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 2. Save the exact list of feature columns
model_features = list(X.columns)
with open('model_features.pkl', 'wb') as f:
    pickle.dump(model_features, f)

print("Files saved successfully using pickle!")



#### 5.Classification :
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#### 1. Load data
df = pd.read_csv('cleaned_charging_records.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])

#### 2. Features
df['StartHour'] = df['StartDatetime'].dt.hour
df['DayOfWeek'] = df['StartDatetime'].dt.dayofweek
df['Month'] = df['StartDatetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

df['Hour_sin'] = np.sin(2 * np.pi * df['StartHour'] / 24.0)
df['Hour_cos'] = np.cos(2 * np.pi * df['StartHour'] / 24.0)
df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

#### 3. Drop leakage & unneeded columns
leakage_cols = ['Duration', 'EndDatetime', 'EndDay', 'EndTime']
unneeded_cols = ['UserID', 'StartDatetime', 'StartDay', 'StartTime', 'StartHour', 'DayOfWeek']
df_model = df.drop(columns=leakage_cols + unneeded_cols)

#### 4. Encode Features
df_encoded = pd.get_dummies(df_model, columns=['Location', 'ChargerCompany', 'ChargerType'], drop_first=True)
X = df_encoded.drop(columns=['Demand'])

#### 5. Create Classification Target (High vs Normal Demand)
median_demand = df_encoded['Demand'].median()
print(f"Median Demand Threshold: {median_demand:.2f}")

# 1 = High Demand (>= median), 0 = Normal Demand (< median)
y = (df_encoded['Demand'] >= median_demand).astype(int)

# Check class distribution
print("\n--- Class Counts ---")
print(y.value_counts())
print("\n--- Class Proportions (%) ---")
print((y.value_counts(normalize=True) * 100).round(2))

#### 6. Stratified Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape:  {X_test.shape}")

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
classification_result={}

###define a function to compute ,store and display clsiification metrics

def evaluate_classifier(name,y_true, y_pred, y_proba =None):
    acc = accuracy_score(y_true, y_pred)            #### Calculate Classification matrics
    prec = precision_score(y_true, y_pred,zero_division =0 )
    rec = recall_score(y_true, y_pred,zero_division =0 )
    f1 = f1_score(y_true, y_pred,zero_division =0)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0

    classification_result[name] = {
        'Accuracy': round(acc,4),
        'Precision':round(prec,4),
        'Recall': round(rec, 4),
        'F1_score': round(f1, 4),
        'ROU -AUC':round(roc, 4)
                }
    print(f"____{name}____")
    print(f"Accuarcy:{acc:4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}\n")

    ### Classification 1.Logistic & Standardscaler

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Scaling features allows lbfgs to converge cleanly
log_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
log_pipeline.fit(X_train, y_train)

y_pred_log = log_pipeline.predict(X_test)
y_proba_log = log_pipeline.predict_proba(X_test)[:, 1]

evaluate_classifier("Logistic Regression", y_test, y_pred_log, y_proba_log)

### Classification - 2.Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_clf.fit(X_train, y_train)

y_pred_dt = dt_clf.predict(X_test)
y_proba_dt = dt_clf.predict_proba(X_test)[:, 1]

evaluate_classifier("Decision Tree", y_test, y_pred_dt, y_proba_dt)
#### Classification - 3.Random Forest 

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100,max_depth= 12, random_state=42)
rf_clf.fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:,1]

evaluate_classifier("Random Forest", y_test,y_pred_rf,y_proba_rf)
## Classification -4.K-Nearest Neighbors(KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7 ,n_jobs=1))
knn_pipeline.fit(X_train,y_train)

y_pred_knn = knn_pipeline.predict(X_test)
y_proba_knn = knn_pipeline.predict_proba(X_test)[:,1]

evaluate_classifier("KNN", y_test,y_pred_knn, y_proba_knn)
### Classification - 5.SVM-Suppot Vector Machine
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

base_svc = LinearSVC(dual=False, random_state=42, max_iter=2000)
svm_pipeline = make_pipeline(StandardScaler(), CalibratedClassifierCV(base_svc))

svm_pipeline.fit(X_train, y_train)

y_pred_svm = svm_pipeline.predict(X_test)
y_proba_svm = svm_pipeline.predict_proba(X_test)[:, 1]

evaluate_classifier("SVM", y_test, y_pred_svm, y_proba_svm)

#### Classification - 6.XGBoost 
### Model 6: XGBoost Classifier
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

evaluate_classifier("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
### Model 7: LightGBM Classifier
from lightgbm import LGBMClassifier

lgb_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

lgb_clf.fit(X_train, y_train)

y_pred_lgb = lgb_clf.predict(X_test)
y_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

evaluate_classifier("LightGBM", y_test, y_pred_lgb, y_proba_lgb)### Clasiification - 6.Light GBM 
### Step 5.8: Classification Models Leaderboard
import pandas as pd

# Reconstruct the results dictionary with your exact verified model metrics
classification_results = {
    'Logistic Regression': {
        'Accuracy': 0.6330, 'Precision': 0.6303, 'Recall': 0.6436, 'F1-Score': 0.6369, 'ROC-AUC': 0.6833
    },
    'Decision Tree': {
        'Accuracy': 0.6671, 'Precision': 0.6512, 'Recall': 0.7199, 'F1-Score': 0.6838, 'ROC-AUC': 0.7257
    },
    'Random Forest': {
        'Accuracy': 0.6763, 'Precision': 0.6617, 'Recall': 0.7213, 'F1-Score': 0.6902, 'ROC-AUC': 0.7439
    },
    'KNN': {
        'Accuracy': 0.6379, 'Precision': 0.6325, 'Recall': 0.6586, 'F1-Score': 0.6453, 'ROC-AUC': 0.6843
    },
    'SVM': {
        'Accuracy': 0.6330, 'Precision': 0.6306, 'Recall': 0.6423, 'F1-Score': 0.6364, 'ROC-AUC': 0.6834
    },
    'XGBoost': {
        'Accuracy': 0.6934, 'Precision': 0.6779, 'Recall': 0.7370, 'F1-Score': 0.7062, 'ROC-AUC': 0.7625
    },
    'LightGBM': {
        'Accuracy': 0.6901, 'Precision': 0.6748, 'Recall': 0.7339, 'F1-Score': 0.7031, 'ROC-AUC': 0.7582
    }
}

# Create and sort the leaderboard
comparison_df = pd.DataFrame(classification_results).T
comparison_df = comparison_df.sort_values(by='ROC-AUC', ascending=False)

print("=== Classification Model Comparison Leaderboard ===")
comparison_df


### Best Model is XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'High'], yticklabels=['Normal', 'High'])
axes[0].set_title('Confusion Matrix - XGBoost')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = 0.7625)')
axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[1].set_title('ROC Curve - XGBoost')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

# 3. Top 10 Feature Importances
importances = pd.Series(xgb_clf.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind='barh', ax=axes[2], color='teal')
axes[2].set_title('Top 10 Feature Importances - XGBoost')
axes[2].set_xlabel('Relative Importance Score')

plt.tight_layout()
plt.show()

### Save Model as Pickle file
import pickle

# 1. Save the trained XGBoost model
model_filename = "best_ev_classification_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(xgb_clf, file)

# 2. Save the feature column order to ensure matching inputs during inference
features_filename = "classification_model_features.pkl"
with open(features_filename, "wb") as file:
    pickle.dump(list(X.columns), file)

print(f"Model saved successfully as: '{model_filename}'")
print(f"Features list saved successfully as: '{features_filename}'")
