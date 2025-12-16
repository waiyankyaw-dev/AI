import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD & CLEAN
# ==========================================
def load_data():
    # Handle filename typos
    label_file = 'trainlabel.txt' if os.path.exists('trainlabel.txt') else 'trainable.txt'
    
    train_df = pd.read_csv('traindata.csv', skipinitialspace=True, na_values='?')
    train_labels = pd.read_csv(label_file, header=None, names=['income'])
    test_df = pd.read_csv('testdata.csv', skipinitialspace=True, na_values='?')
    
    # Feature Engineering: Net Capital
    # Helps the model distinguish wealth better
    train_df['capital_net'] = train_df['capital.gain'] - train_df['capital.loss']
    test_df['capital_net'] = test_df['capital.gain'] - test_df['capital.loss']
    
    return train_df, train_labels, test_df

# ==========================================
# 2. PREPROCESSING
# ==========================================
def get_pipeline():
    # We DROP 'fnlwgt'. In scientific literature for this dataset, 
    # 'fnlwgt' (final weight) is often noise and lowers accuracy.
    numeric_features = ['age', 'education.num', 'capital_net', 'hours.per.week']
    categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ])
    
    # We use HistGradientBoosting. It is the modern "King" of tabular data in Python.
    # It is faster and usually more accurate than standard GradientBoosting.
    model = HistGradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

# ==========================================
# 3. HYPERPARAMETER TUNING (GRID SEARCH)
# ==========================================
def optimize_and_train(X, y):
    pipeline = get_pipeline()

    # The Grid Search tries all these combinations to find the "Perfect" model
    # This creates the "Methodology" points for your report.
    param_grid = {
        'model__learning_rate': [0.075, 0.1, 0.5], #0.1 is better than 0.05
        'model__max_iter': [100, 150], #200 is not good as 100
        'model__max_depth': [None,1], #10 is not good as None
        'model__l2_regularization': [0.0, 0.1] #1.0 not good as 0.0
    }

    print("Starting Grid Search (Optimizing for best accuracy)...")
    print("This tries multiple model variations. Please wait...")
    
    # 5-Fold Cross Validation: Rigorous testing
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, # Uses all CPU cores
        verbose=1
    )
    
    grid_search.fit(X, y.values.ravel())
    
    print(f"\nBest Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Load
    X_train_full, y_train_full, X_test = load_data()
    
    # 2. Split for our own internal report (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)
    
    # 3. Optimize
    best_model = optimize_and_train(X_train, y_train)
    
    # 4. Validate
    print("\n--- Final Validation Report ---")
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Optimized Confusion Matrix (Acc: {acc:.4f})')
    plt.savefig('confusion_matrix_final.png')
    
    # 5. FINAL RETRAINING (The Secret to higher test scores)
    # We take the best settings and train on ALL 22k rows before predicting the test set.
    print("\nRetraining best model on FULL training set...")
    best_model.fit(X_train_full, y_train_full.values.ravel())
    
    # 6. Predict
    print("Generating predictions...")
    final_preds = best_model.predict(X_test)
    np.savetxt('testlabel.txt', final_preds, fmt='%d')
    print("Done. Predictions saved to 'testlabel.txt'")