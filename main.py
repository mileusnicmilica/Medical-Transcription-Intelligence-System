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

    # 5. PIPELINE: TF-IDF + LinearSVC, with GridSearchCV tuning (kept consistent with the
    # no-Surgery variant in step 13, so the Surgery-removal comparison isolates that one
    # variable rather than conflating it with hyperparameter tuning)
    base_pipeline = SklearnPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=5000))
    ])

    param_grid = {'clf__C': [0.1, 1, 10]}

    # 6. CROSS-VALIDATION (on the base, untuned pipeline - kept as a separate diagnostic,
    # not the basis for the final reported numbers)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        base_pipeline, X_train, y_train, cv=cv,
        scoring=['f1_weighted', 'balanced_accuracy']
    )

    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(
        f"Weighted F1:        {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
    print(
        f"Balanced Accuracy:  {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")

    # 7. GRIDSEARCHCV TUNING + FINAL FIT - TRAIN SET and EVALUATION - TEST SET
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(base_pipeline, param_grid, cv=3, scoring='f1_macro')
    grid.fit(X_train, y_train)
    pipeline = grid.best_estimator_
    print(f"Best hyperparameters (with Surgery): {grid.best_params_}")

    predictions = pipeline.predict(X_test)

    weighted_f1, macro_f1, balanced_acc = evaluate_model(y_test, predictions)

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
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }

    print("\n=== FINAL TEST SET RESULTS ===")
    results_df = pd.DataFrame(all_results).T
    print(results_df)

    # 10. SETFIT - fair ablation: both variants trained from scratch, same num_iterations
    from src.setfit_classifier import (
        evaluate_setfit, prepare_setfit_data,
        train_setfit, save_setfit_model
    )

    print("\n=== SETFIT FEW-SHOT CLASSIFICATION (raw vs. lemmatized ablation) ===")

    print("\n=== SETFIT (LEMMATIZED, TRAINING FROM SCRATCH) ===")
    train_ds_lemma, val_ds_lemma, test_ds_lemma, le_lemma = prepare_setfit_data(
        df, n_samples=16, text_column='cleaned_transcription'
    )
    setfit_model_lemma = train_setfit(train_ds_lemma, val_ds_lemma, le_lemma, num_iterations=1)
    save_setfit_model(setfit_model_lemma, le_lemma, name="setfit_v1_lemmatized")
    setfit_f1_lemma, setfit_macro_f1_lemma, setfit_balanced_acc_lemma = evaluate_setfit(setfit_model_lemma,
                                                                                        test_ds_lemma, le_lemma)
    all_results['SetFit (lemmatized)'] = {'weighted_f1': setfit_f1_lemma, 'macro_f1': setfit_macro_f1_lemma,
                                          'balanced_accuracy': setfit_balanced_acc_lemma}

    print("\n=== SETFIT (RAW TEXT, TRAINING FROM SCRATCH) ===")
    train_ds_raw, val_ds_raw, test_ds_raw, le_raw = prepare_setfit_data(
        df, n_samples=16, text_column='transcription'
    )
    setfit_model_raw = train_setfit(train_ds_raw, val_ds_raw, le_raw, num_iterations=1)
    save_setfit_model(setfit_model_raw, le_raw, name="setfit_v1_raw")
    setfit_f1_raw, setfit_macro_f1_raw, setfit_balanced_acc_raw = evaluate_setfit(setfit_model_raw, test_ds_raw, le_raw)
    all_results['SetFit (raw)'] = {'weighted_f1': setfit_f1_raw, 'macro_f1': setfit_macro_f1_raw,
                                   'balanced_accuracy': setfit_balanced_acc_raw}
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

    # 12. ENTITY EXTRACTION (Ollama - Local LLM) + JUDGE COMPARISON (single Groq vs. heterogeneous panel)
    try:
        import requests
        requests.get("http://localhost:11434", timeout=2)
        from src.extractor import run_extraction_pipeline

        if not os.environ.get("GROQ_API_KEY"):
            print("\nGROQ_API_KEY not set — falling back to local Ollama judge only.")
            extraction_results = run_extraction_pipeline(df, n_samples=5, judge="ollama")
        else:
            print("\n=== ENTITY EXTRACTION (Ollama) + JUDGE: GROQ (single, stronger judge) ===")
            extraction_results_groq = run_extraction_pipeline(df, n_samples=5, judge="groq")

            print("\n=== ENTITY EXTRACTION (Ollama) + JUDGE: PANEL (heterogeneous + self-eval baseline) ===")
            extraction_results_panel = run_extraction_pipeline(df, n_samples=5, judge="panel")

            groq_scores = [r['evaluation'].get('overall_score') for r in extraction_results_groq
                           if isinstance(r['evaluation'].get('overall_score'), (int, float))]
            panel_means = [r['evaluation']['panel_summary'].get('panel_mean') for r in extraction_results_panel
                           if r['evaluation'].get('panel_summary')
                           and isinstance(r['evaluation']['panel_summary'].get('panel_mean'), (int, float))]
            self_eval_scores = [r['evaluation']['panel_summary'].get('self_eval_score') for r in
                                extraction_results_panel
                                if r['evaluation'].get('panel_summary')
                                and isinstance(r['evaluation']['panel_summary'].get('self_eval_score'), (int, float))]

            print("\n=== JUDGE METHODOLOGY COMPARISON (same 5 samples, random_state=42) ===")
            judge_comparison = pd.DataFrame({
                'avg_score': {
                    'Self-eval baseline (Ollama judges itself)': sum(self_eval_scores) / len(
                        self_eval_scores) if self_eval_scores else None,
                    'Single Groq judge (openai/gpt-oss-120b)': sum(groq_scores) / len(
                        groq_scores) if groq_scores else None,
                    'Heterogeneous panel (gpt-oss-120b + qwen3.6-27b, mean)': sum(panel_means) / len(
                        panel_means) if panel_means else None,
                }
            })
            print(judge_comparison)

            extraction_results = extraction_results_panel  # keep the richest result set, in case it's needed later
    except Exception as e:
        print(f"\n=== Ollama is not available: {e} ===")
        print("Start 'ollama serve' first and then run again.")




    print("\n=== FINAL COMPARISON TABLE ===")
    results_df = pd.DataFrame(all_results).T
    print(results_df)
    # 13. SURGERY CLASS REMOVAL - COMPARATIVE ANALYSIS + GridSearchCV TUNING
    # (macro-F1 is the key metric, since Surgery's dominance specifically distorts
    # macro-averaged performance; these tuned, Surgery-removed models become the
    # final system used in the Streamlit demo)
    from sklearn.model_selection import GridSearchCV

    print("\n=== SURGERY CLASS REMOVAL - COMPARATIVE ANALYSIS ===")

    df_no_surgery = df[df['medical_specialty'] != 'Surgery'].reset_index(drop=True)
    print(f"Removed Surgery class: {len(df)} -> {len(df_no_surgery)} samples "
          f"({len(df) - len(df_no_surgery)} samples removed)")

    surgery_results = {}

    # --- TF-IDF + LinearSVC, without Surgery, with GridSearchCV hyperparameter tuning ---
    X_ns = df_no_surgery['cleaned_transcription']
    y_ns = df_no_surgery['medical_specialty']

    X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
        X_ns, y_ns, test_size=0.2, random_state=42, stratify=y_ns
    )

    base_pipeline_ns = SklearnPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])

    # GridSearchCV over the full pipeline, scored on macro-F1 since that's our key metric
    param_grid = {'clf__C': [0.1, 1, 10]}
    grid_ns = GridSearchCV(base_pipeline_ns, param_grid, cv=3, scoring='f1_macro')
    grid_ns.fit(X_train_ns, y_train_ns)
    pipeline_ns = grid_ns.best_estimator_
    print(f"Best hyperparameters (no Surgery): {grid_ns.best_params_}")

    predictions_ns = pipeline_ns.predict(X_test_ns)

    print("\n--- TF-IDF + LinearSVC (no Surgery, GridSearch-tuned) ---")
    tfidf_w_f1_ns, tfidf_macro_f1_ns, tfidf_bal_acc_ns = evaluate_model(y_test_ns, predictions_ns)
    surgery_results['TF-IDF + LinearSVC (no Surgery, tuned)'] = {
        'weighted_f1': tfidf_w_f1_ns, 'macro_f1': tfidf_macro_f1_ns, 'balanced_accuracy': tfidf_bal_acc_ns
    }

    # Save this as the FINAL TF-IDF model used in the Streamlit demo
    save_artifacts(pipeline_ns, name="linear_svc_v1_no_surgery")

    # --- SetFit (lemmatized), without Surgery - same training budget as the raw/lemmatized ablation ---
    print("\n--- SetFit (lemmatized, no Surgery) ---")
    train_ds_ns, val_ds_ns, test_ds_ns, le_ns = prepare_setfit_data(
        df_no_surgery, n_samples=16, text_column='cleaned_transcription'
    )
    setfit_model_ns = train_setfit(train_ds_ns, val_ds_ns, le_ns, num_iterations=1)
    save_setfit_model(setfit_model_ns, le_ns, name="setfit_v1_lemmatized_no_surgery")
    setfit_w_f1_ns, setfit_macro_f1_ns, setfit_bal_acc_ns = evaluate_setfit(setfit_model_ns, test_ds_ns, le_ns)
    surgery_results['SetFit (lemmatized, no Surgery)'] = {
        'weighted_f1': setfit_w_f1_ns, 'macro_f1': setfit_macro_f1_ns, 'balanced_accuracy': setfit_bal_acc_ns
    }

    # --- Before/after comparison table ---
    print("\n=== SURGERY REMOVAL - BEFORE/AFTER COMPARISON (focus on macro_f1) ===")
    surgery_comparison = pd.DataFrame({
        'TF-IDF + LinearSVC (with Surgery)': all_results.get('TF-IDF + LinearSVC', {}),
        'TF-IDF + LinearSVC (no Surgery, tuned)': surgery_results.get('TF-IDF + LinearSVC (no Surgery, tuned)', {}),
        'SetFit (with Surgery, lemmatized)': all_results.get('SetFit (lemmatized)', {}),
        'SetFit (no Surgery, lemmatized)': surgery_results.get('SetFit (lemmatized, no Surgery)', {}),
    }).T
    print(surgery_comparison)


if __name__ == "__main__":
    main()