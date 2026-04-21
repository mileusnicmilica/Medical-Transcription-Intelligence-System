# src/classifiers.py
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def train_baseline_svc(X_train, y_train):
    """
    Trains a LinearSVC model using GridSearchCV for hyperparameter tuning.
    This serves as our traditional NLP baseline.
    """
    print("--- Training Baseline LinearSVC ---")
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid, cv=3)
    grid.fit(X_train, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_