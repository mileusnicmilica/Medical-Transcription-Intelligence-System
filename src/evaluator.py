# src/evaluator.py
from sklearn.metrics import f1_score, accuracy_score, classification_report


def evaluate_model(y_true, y_pred):
    """
    Calculates metrics and returns them as a dictionary.
    """
    print("--- Model Evaluation ---")

    # Calculate specific scores
    w_f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    # Print the full report to console
    print(classification_report(y_true, y_pred))
    print(f"Final Weighted F1 Score: {w_f1:.4f}")

    return w_f1, acc