# src/setfit_classifier.py
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
import joblib
import os


def prepare_setfit_data(df, n_samples=16):
    """
    Prepares few-shot dataset for SetFit.
    Takes n_samples per class from training data.

    Args:
        df: DataFrame with 'cleaned_transcription' and 'medical_specialty'
        n_samples: number of samples per class (8 or 16 as per project proposal)

    Returns:
        train_dataset, test_dataset, label_encoder
    """
    # Encode string labels to integers (SetFit requires numeric labels)
    le = LabelEncoder()
    df = df.copy()
    df['label'] = le.fit_transform(df['medical_specialty'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_transcription'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    # Few-shot: sample n_samples per class
    few_shot_samples = []
    for label_val in train_df['label'].unique():
        subset = train_df[train_df['label'] == label_val]
        sampled = subset.sample(min(n_samples, len(subset)), random_state=42)
        few_shot_samples.append(sampled)

    few_shot_train = pd.concat(few_shot_samples).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(few_shot_train)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset, le


def train_setfit(train_dataset, test_dataset, label_encoder,
                 model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """
    Trains a SetFit model.
    Uses paraphrase-MiniLM-L6-v2 as base (lighter than PubMedBERT, good for CPU).
    """
    num_classes = len(label_encoder.classes_)

    model = SetFitModel.from_pretrained(model_name)

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="f1",
        metric_kwargs={"average": "weighted"}
    )

    trainer.train()

    return model


def evaluate_setfit(model, test_dataset, label_encoder):
    """
    Evaluates SetFit model on test set.
    """
    texts = test_dataset['text']
    true_labels = test_dataset['label']

    predictions = model.predict(texts)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    balanced_acc = balanced_accuracy_score(true_labels, predictions)

    print("\n--- SetFit Model Evaluation ---")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(true_labels, predictions,
                                target_names=label_encoder.classes_))

    return weighted_f1, balanced_acc


def save_setfit_model(model, label_encoder, name="setfit_v1"):
    """
    Saves SetFit model and label encoder.
    """
    os.makedirs("models", exist_ok=True)
    model.save_pretrained(f"models/{name}")
    joblib.dump(label_encoder, f"models/{name}_label_encoder.pkl")
    print(f"\n--- Success: SetFit model saved as '{name}' ---")

def load_setfit_model(name="setfit_v1"):
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(f"models/{name}")
    le = joblib.load(f"models/{name}_label_encoder.pkl")
    return model, le