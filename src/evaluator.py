# src/evaluator.py
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report


def evaluate_model(y_true, y_pred):
    """
    Calculates metrics and returns them as a dictionary.
    """
    print("--- Model Evaluation ---")

    w_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')  # NEW - critical for class-imbalance ablations (e.g. Surgery removal)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"Final Weighted F1 Score: {w_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))

    return w_f1, macro_f1, balanced_acc
