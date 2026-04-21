# src/balancing.py
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def balance_data(df, text_column='cleaned_transcription', target_column='medical_specialty'):
    """
    Balances the dataset using SMOTE.
    SMOTE requires numerical input so we apply TF-IDF vectorization first.
    """
    print(f"Balancing classes using SMOTE for: {target_column}")

    # 1. Vectorize text (SMOTE cannot work directly on strings)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df[text_column])
    y = df[target_column]

    # 2. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"Resampling complete. New shape: {X_resampled.shape}")
    return X_resampled, y_resampled, tfidf