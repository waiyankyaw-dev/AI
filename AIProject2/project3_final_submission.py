import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Scikit-Learn Ecosystem
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data():
    print("Loading Data...")
    label_file = 'trainlabel.txt' if os.path.exists('trainlabel.txt') else 'trainable.txt'
    
    train_df = pd.read_csv('traindata.csv', skipinitialspace=True, na_values='?')
    train_labels = pd.read_csv(label_file, header=None, names=['income'])
    test_df = pd.read_csv('testdata.csv', skipinitialspace=True, na_values='?')
    
    return train_df, train_labels, test_df

# ==========================================
# 2. DATA PREPROCESSING (Baseline Strategy)
# ==========================================
def get_preprocessor():
    """
    Returns the Baseline ColumnTransformer.
    - Imputes missing values.
    - Scales numerical features.
    - One-Hot encodes categorical features.
    - Keeps all original columns (including fnlwgt).
    """
    numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    
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
    
    return preprocessor

# ==========================================
# 3. DEFINING MODEL SEARCH SPACE
# ==========================================
def get_model_experiments():
    """
    Returns a list of dictionaries containing:
    - Name
    - Model Object
    - Hyperparameter Grid
    """
    experiments = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'model__C': [0.1, 1.0, 10],
                'model__solver': ['lbfgs', 'liblinear']
            }
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'model__max_depth': [10, 20, None],
                'model__min_samples_leaf': [2, 10],
                'model__criterion': ['gini', 'entropy']
            }
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [15, 25],
                'model__min_samples_split': [5, 10]
            }
        },
        {
            'name': 'SVM (RBF Kernel)',
            'model': SVC(random_state=42),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf'], 
                'model__gamma': ['scale', 'auto']
            }
        },
        {
            'name': 'KNN',
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [5, 9, 15],
                'model__weights': ['uniform', 'distance']
            }
        },
        {
            'name': 'Neural Network (MLP)',
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {
                'model__hidden_layer_sizes': [(50,), (100, 50)],
                'model__activation': ['tanh', 'relu'],
                'model__alpha': [0.0001, 0.05]
            }
        },
        # Classic Gradient Boosting
        {
            'name': 'Gradient Boosting (Classic)',
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            }
        },
        # Modern Gradient Boosting
        {
            'name': 'Hist Gradient Boosting',
            'model': HistGradientBoostingClassifier(random_state=42),
            'params': {
                'model__learning_rate': [0.05, 0.1],
                'model__max_iter': [100, 200],
                'model__l2_regularization': [0.0, 1.0],
                'model__max_depth': [None, 10]
            }
        }
    ]
    return experiments

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    start_total = time.time()
    
    # 1. Load Data
    train_df, train_labels, test_df = load_data()
    X = train_df
    y = train_labels['income']

    # 2. Split (Hold-out set for final verification)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Setup Experiments
    preprocessor = get_preprocessor()
    experiments = get_model_experiments()
    
    results = []

    print(f"\nStarting Final Model Benchmarking...")
    print(f"Testing {len(experiments)} Models with Grid Search Cross-Validation.")
    print("=" * 80)

    best_overall_score = 0
    best_overall_pipeline = None
    best_overall_name = ""

    # --- THE LOOP ---
    for exp in experiments:
        model_name = exp['name']
        print(f"Optimizing {model_name}...", end=" ")
        
        # Construct Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', exp['model'])
        ])
        
        # Grid Search
        grid = GridSearchCV(
            pipeline,
            exp['params'],
            cv=4,             # 4-Fold Cross Validation
            scoring='accuracy',
            n_jobs=-1,        # Use all CPU cores
            verbose=0
        )
        
        grid.fit(X_train, y_train)
        
        # Validate on Hold-out set
        val_acc = grid.score(X_val, y_val)
        
        print(f"Done. Best CV: {grid.best_score_:.4f} | Validation: {val_acc:.4f}")
        
        # Store Result
        results.append({
            'Model': model_name,
            'Best CV Accuracy': grid.best_score_,
            'Validation Accuracy': val_acc,
            'Best Params': grid.best_params_
        })
        
        # Check if this is the new champion
        if val_acc > best_overall_score:
            best_overall_score = val_acc
            best_overall_pipeline = grid.best_estimator_
            best_overall_name = model_name

    # ==========================================
    # 5. RESULTS ANALYSIS
    # ==========================================
    print("\n" + "="*80)
    print("FINAL LEADERBOARD")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Validation Accuracy', ascending=False)
    
    print(results_df[['Model', 'Validation Accuracy', 'Best CV Accuracy']].to_string(index=False))
    
    print(f"\nCHAMPION MODEL: {best_overall_name} with Accuracy: {best_overall_score:.4f}")
    
    # ==========================================
    # 6. FINAL REPORTING & PREDICTION
    # ==========================================
    print("\nGenerating assets for report...")
    
    # Confusion Matrix for the Winner
    y_pred = best_overall_pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix - {best_overall_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_final.png')
    print("Saved 'confusion_matrix_final.png'")
    
    # Classification Report
    print("\nDetailed Report for Winner:")
    print(classification_report(y_val, y_pred))
    
    # Final Retraining on Full Data
    print(f"Retraining {best_overall_name} on 100% of Training Data...")
    best_overall_pipeline.fit(X, y)
    
    # Predictions
    print("Predicting Test Data...")
    test_preds = best_overall_pipeline.predict(test_df)
    np.savetxt('testlabel.txt', test_preds, fmt='%d')
    print("Predictions saved to 'testlabel.txt'.")
    
    end_time = time.time()
    print(f"\nTotal Execution Time: {(end_time - start_total)/60:.2f} minutes")