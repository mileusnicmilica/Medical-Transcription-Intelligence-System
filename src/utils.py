import joblib
import os


def save_artifacts(pipeline, name="baseline_svc"):
    """
    Saves the trained pipeline (includes TF-IDF + SMOTE + model) to the 'models' directory.
    """
    os.makedirs('models', exist_ok=True)

    pipeline_path = f"models/{name}_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    print(f"\n--- Success: Pipeline saved as '{name}' ---")


def load_artifacts(name="baseline_svc"):
    pipeline_path = f"models/{name}_pipeline.pkl"
    pipeline = joblib.load(pipeline_path)
    return pipeline

