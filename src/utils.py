import joblib
import os


def save_artifacts(model, vectorizer, name="baseline_svc"):
    """
    Saves the trained model and TF-IDF vectorizer to the 'models' directory.
    """
    if not os.path.exists('models'):
        os.makedirs('models')

    model_path = f"models/{name}_model.pkl"
    vectorizer_path = f"models/{name}_tfidf.pkl"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\n--- Success: Model and Vectorizer saved as '{name}' ---")


def load_artifacts(name="baseline_svc"):
    """
    Loads saved model and vectorizer from the disk.
    """
    model_path = f"models/{name}_model.pkl"
    vectorizer_path = f"models/{name}_tfidf.pkl"

    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    return model, tfidf

