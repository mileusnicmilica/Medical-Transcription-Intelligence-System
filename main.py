# main.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.data_loader import download_and_load_data, clean_medical_data
from src.preprocessor import MedicalPreprocessor
from src.visualizer import plot_class_distribution
from src.evaluator import evaluate_model
from src.utils import save_artifacts

def main():

    all_results = {}

    # 1. LOAD & BASIC CLEAN
    df = download_and_load_data()
    df = clean_medical_data(df)

    # 2. VISUALIZATION BEFORE BALANCING
    plot_class_distribution(df, filename='class_distribution_before_balancing')

    # 3. SPACY PREPROCESSING
    cache_path = "data/cleaned_data_cache.csv"

    if os.path.exists(cache_path):
        print("Loading from cache, skipping preprocessing...")
        df = pd.read_csv(cache_path)
    else:
        processor = MedicalPreprocessor()
        df = processor.preprocess_dataframe(df)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_path, index=False)
        print("Preprocessing saved to cache.")

    # 4. TRAIN/TEST SPLIT
    X = df['cleaned_transcription']
    y = df['medical_specialty']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. PIPELINE: TF-IDF + SMOTE + LinearSVC
    pipeline = SklearnPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])

    # 6. CROSS-VALIDATION
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['f1_weighted', 'balanced_accuracy']
    )

    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Weighted F1:        {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
    print(f"Balanced Accuracy:  {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")

    # 7. FINAL FIT - TRAIN SET and EVALUATION - TEST SET
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    weighted_f1, balanced_acc = evaluate_model(y_test, predictions)

    # VISUALIZATION AFTER BALANCING
    from imblearn.over_sampling import RandomOverSampler as ROS
    import numpy as np

    ros = ROS(random_state=42)
    X_dummy = np.zeros((len(X_train), 1))
    _, y_balanced_viz = ros.fit_resample(X_dummy, y_train)
    balanced_df = pd.DataFrame({'medical_specialty': y_balanced_viz})
    plot_class_distribution(
        balanced_df,
        title=f'Class Distribution AFTER Balancing (RandomOverSampler)\nAll classes equalized to majority class size',
        filename='class_distribution_after_balancing'
    )

    # 8. SAVE
    save_artifacts(pipeline, name="linear_svc_v1")

    # 9. FINAL COMPARISON TABLE

    all_results['TF-IDF + LinearSVC'] = {
        'weighted_f1': weighted_f1,
        'balanced_accuracy': balanced_acc
    }

    print("\n=== FINAL TEST SET RESULTS ===")
    results_df = pd.DataFrame(all_results).T
    print(results_df)

    # 10. SETFIT - loading already trained model
    from src.setfit_classifier import load_setfit_model, evaluate_setfit, prepare_setfit_data

    print("\n=== SETFIT FEW-SHOT CLASSIFICATION ===")
    from src.setfit_classifier import (
        load_setfit_model, evaluate_setfit, prepare_setfit_data,
        train_setfit, save_setfit_model
    )

    print("\n=== SETFIT (LEMMATIZED, EXISTING MODEL) ===")
    _, _, test_dataset, label_encoder = prepare_setfit_data(df, n_samples=16, text_column='cleaned_transcription')
    setfit_model, label_encoder = load_setfit_model()
    setfit_f1, setfit_balanced_acc = evaluate_setfit(setfit_model, test_dataset, label_encoder)
    all_results['SetFit (lemmatized)'] = {'weighted_f1': setfit_f1, 'balanced_accuracy': setfit_balanced_acc}

    print("\n=== SETFIT (RAW TEXT, ABLATION - TRAINING FROM SCRATCH) ===")
    train_ds_raw, val_ds_raw, test_ds_raw, le_raw = prepare_setfit_data(df, n_samples=16, text_column='transcription')
    setfit_model_raw = train_setfit(train_ds_raw, val_ds_raw, le_raw, num_iterations=1)
    save_setfit_model(setfit_model_raw, le_raw, name="setfit_v1_raw")
    setfit_f1_raw, setfit_balanced_acc_raw = evaluate_setfit(setfit_model_raw, test_ds_raw, le_raw)
    all_results['SetFit (raw)'] = {'weighted_f1': setfit_f1_raw, 'balanced_accuracy': setfit_balanced_acc_raw}

    # 11. SEMANTIC SEARCH - FAISS (comparison of embedding models)
    from src.searcher import MedicalSearcher, evaluate_search

    print("\n=== SEMANTIC SEARCH (FAISS) ===")

    embedding_models = [
        "all-MiniLM-L6-v2",
        "NeuML/pubmedbert-base-embeddings"
    ]
    text_columns = ["transcription", "cleaned_transcription"]  # raw vs. lemmatized

    search_results_all = {}

    for model_name in embedding_models:
        for text_column in text_columns:
            label = f"{model_name} [{text_column}]"
            print(f"\n--- Model: {model_name} | Tekst: {text_column} ---")
            searcher = MedicalSearcher(model_name=model_name)

            index_path = f"data/faiss_index_{model_name.replace('/', '_')}_{text_column}"

            if os.path.exists(f"{index_path}/index.faiss"):
                print("Loading existing index...")
                searcher.load(index_path)
            else:
                print("Building index...")
                searcher.build_index(df, text_column=text_column)
                searcher.save(index_path)

            print("\n--- Search Evaluation ---")
            results = evaluate_search(searcher, df, k_values=[1, 3, 5], n_queries=100, text_column=text_column)
            search_results_all[label] = results

    print("\n=== EMBEDDING MODEL COMPARISON (raw vs. lemmatized) ===")
    comparison_df = pd.DataFrame(search_results_all).T
    print(comparison_df)
    # 12. ENTITY EXTRACTION (Ollama - Local LLM)
    try:
        import requests
        requests.get("http://localhost:11434", timeout=2)
        from src.extractor import run_extraction_pipeline
        print("\n=== ENTITY EXTRACTION (Ollama) + JUDGE (Groq) ===")
        if os.environ.get("GROQ_API_KEY"):
            extraction_results = run_extraction_pipeline(df, n_samples=5, judge="groq")
        else:
            print("GROQ_API_KEY — fallback on local Ollama judge.")
            extraction_results = run_extraction_pipeline(df, n_samples=5, judge="ollama")
    except Exception as e:
        print(f"\n=== Ollama is not available: {e} ===")
        print("Start 'ollama serve' first and then run again.")


    print("\n=== FINAL COMPARISON TABLE ===")
    results_df = pd.DataFrame(all_results).T
    print(results_df)

if __name__ == "__main__":
    main()