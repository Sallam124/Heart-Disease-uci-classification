
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    df = pd.read_csv(r'c:\Users\salla\OneDrive\Desktop\ML Project\Dataset_1\dataset\heart_disease.csv', na_values=['NA', 'na', 'N/A', 'n/a', ''])
    df.columns = df.columns.str.strip()
    target_col = 'Heart_ stroke'
    if target_col not in df.columns:
        # Fallback if column name is different
        print(f"Columns: {df.columns}")
        target_col = [c for c in df.columns if 'heart' in c.lower() and 'stroke' in c.lower()][0]
    
    df['target'] = (df[target_col].str.lower() == 'yes').astype(int)
    df = df.drop(columns=[target_col])
    X = df.drop(columns=['target'])
    y = df['target']

    print(f'Total Samples: {len(df)}')
    print(f'Class Counts:\n{y.value_counts()}')
    print(f'Class Percentages:\n{y.value_counts(normalize=True)}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    categorical_features = ['Gender', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100))
    ])

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print('\n--- Random Forest (Balanced) Results ---')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision (1): {precision_score(y_test, y_pred, zero_division=0):.4f}')
    print(f'Recall (1): {recall_score(y_test, y_pred):.4f}')
    print(f'F1 (1): {f1_score(y_test, y_pred):.4f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred):.4f}')

except Exception as e:
    print(f"Error: {e}")
