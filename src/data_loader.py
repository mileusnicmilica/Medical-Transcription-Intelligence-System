import kagglehub as kagglehub
import pandas as pd
import os

def download_and_load_data():
    """
    Downloads the Medical Transcriptions dataset from Kaggle using kagglehub
    and loads the main CSV file into a Pandas DataFrame.

    Returns:
        pd.DataFrame: Raw dataset containing medical transcriptions.
    """
    # Download the latest version of the dataset
    # it won't re-download if the files exist, kagglehub handles caching
    path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")

    # Construct the full path to the CSV file
    csv_path = os.path.join(path, "mtsamples.csv")
    df = pd.read_csv(csv_path)

    # remove the redundant index column, usually the case with kaggle datasets
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df

def clean_medical_data(df):
    """
    Performs initial data cleaning:
    1. Removes records with missing transcriptions.
    2. Filters out medical specialties with low sample counts to ensure
       statistical significance during training/evaluation.

    Args:
        df (pd.DataFrame): The raw medical transcription dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for preprocessing.
    """
    # Drop rows where 'transcription' is NaN because its critical for NLP tasks
    df = df.dropna(subset=['transcription'])

    # Calculate frequency of each medical specialty
    counts = df['medical_specialty'].value_counts()

    # Keep only categories with at least 20 samples
    # This addresses extreme class imbalance mentioned in the project proposal
    valid_categories = counts[counts >= 20].index
    df = df[df['medical_specialty'].isin(valid_categories)]

    # Optional: Reset index after filtering for a clean sequence
    df = df.reset_index(drop=True)

    return df