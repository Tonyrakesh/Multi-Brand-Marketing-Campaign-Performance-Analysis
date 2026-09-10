import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load Data
df = pd.read_csv('D:\\Python\\Multi-Brand Marketing Campaign Performance Analysis\\all_campaign_data_cleaned.csv')

# Extract Calendar Features
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year.fillna(2025).astype(int)
df['Month'] = df['Date'].dt.month.fillna(6).astype(int)
df['Day'] = df['Date'].dt.day.fillna(15).astype(int)
df['DayOfWeek'] = df['Date'].dt.dayofweek.fillna(2).astype(int)

# Categorical Encodings
channel_cols = [
    'Channel_Email',
    'Channel_Facebook',
    'Channel_Google',
    'Channel_Instagram',
    'Channel_WhatsApp',
    'Channel_YouTube',
]
cat_cols = [
    'Brand',
    'Campaign_Type',
    'Target_Audience',
    'Language',
    'Customer_Segment',
]
cat_dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)

# Feature Sets
# Regression Feature Set (Uses all columns including ROI, CAC, Conversions)
reg_num_cols = [
    'Duration',
    'Impressions',
    'Clicks',
    'Leads',
    'Conversions',
    'Acquisition_Cost',
    'ROI',
    'Engagement_Score',
    'Profit_Loss_Flag',
    'Year',
    'Month',
    'Day',
    'DayOfWeek',
]
X_reg = pd.concat([df[reg_num_cols + channel_cols], cat_dummies], axis=1)
y_reg = df['Revenue']

# Classification Feature Set (Strictly excludes ROI)
cls_num_cols = [
    'Duration',
    'Impressions',
    'Clicks',
    'Leads',
    'Conversions',
    'Revenue',
    'Acquisition_Cost',
    'Engagement_Score',
    'Year',
    'Month',
    'Day',
    'DayOfWeek',
]
X_cls = pd.concat([df[cls_num_cols + channel_cols], cat_dummies], axis=1)
y_cls = df['Profit_Loss_Flag']

# Train Models
print('Training HistGradientBoostingRegressor...')
reg_model = HistGradientBoostingRegressor(
    max_iter=300,
    max_leaf_nodes=63,
    learning_rate=0.08,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
).fit(X_reg, y_reg)

print('Training XGBoostClassifier...')
cls_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
).fit(X_cls, y_cls)

# Export Artifacts
joblib.dump(reg_model, 'reg_model.joblib')
joblib.dump(cls_model, 'cls_model.joblib')
joblib.dump(X_reg.columns.tolist(), 'reg_columns.joblib')
joblib.dump(X_cls.columns.tolist(), 'cls_columns.joblib')
print(
    'Artifacts successfully saved: reg_model.joblib, cls_model.joblib,'
    ' column files.'
)
