# src/balancing.py
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


def balance_data(df, text_column='cleaned_transcription', target_column='medical_specialty'):
    """
    Balances the dataset using SMOTE.
    Performs train/test split first to avoid data leakage,
    then fits TF-IDF only on training data before applying SMOTE.
    """
    print(f"Balancing classes using SMOTE for: {target_column}")

    X = df[text_column]
    y = df[target_column]

    # 1. Split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Fit TF-IDF only on training data
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)  # only transform, never fit on test

    # 3. Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

    print(f"Resampling complete. New shape: {X_resampled.shape}")
    return X_resampled, y_resampled, X_test_tfidf, y_test, tfidf