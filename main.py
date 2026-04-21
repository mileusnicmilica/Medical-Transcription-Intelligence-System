# main.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom modules
from src.data_loader import download_and_load_data, clean_medical_data
from src.preprocessor import MedicalPreprocessor
from src.visualizer import plot_class_distribution
from src.balancing import balance_data
from src.classifiers import train_baseline_svc
from src.evaluator import evaluate_model
from src.utils import save_artifacts

def main():
    # Initialize a dictionary to store all model results for final comparison
    all_results = {}

    # 1. LOAD & BASIC CLEAN
    df = download_and_load_data()
    df = clean_medical_data(df)

    # 2. VISUALIZATION BEFORE BALANCING
    plot_class_distribution(df)

    # 3. SPACY PREPROCESSING
    processor = MedicalPreprocessor()
    df = processor.preprocess_dataframe(df)
    df.to_csv("data/cleaned_data_cache.csv", index=False)
    print("Preprocessing saved to cache.")

    # 4. BALANCING - SMOTE
    X_balanced, y_balanced, tfidf = balance_data(df)

    # 5. TEST/TRAIN SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # 6. TRAINING BASELINE MODEL
    model = train_baseline_svc(X_train, y_train)

    # 8. SAVING THE RESULT (The English version)
    save_artifacts(model, tfidf, name="linear_svc_v1")

    # 7. FINAL RESULTS & EVALUATION
    predictions = model.predict(X_test)


    # function with two outputs
    weighted_f1, accuracy = evaluate_model(y_test, predictions)

    # Saving results for the final table
    all_results['TF-IDF + LinearSVC'] = {
        'weighted_f1': weighted_f1,
        'accuracy': accuracy
    }
    print("\n=== FINAL COMPARISON ===")
    results_df = pd.DataFrame(all_results).T
    print(results_df)


if __name__ == "__main__":
    main()