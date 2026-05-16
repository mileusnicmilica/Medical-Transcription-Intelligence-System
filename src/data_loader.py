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
    df = df.dropna(subset=['transcription'])

    df['medical_specialty'] = df['medical_specialty'].str.strip() #some specialties have values such as " Neurology" so model cannot learn properly

    counts = df['medical_specialty'].value_counts()
    valid_categories = counts[counts >= 20].index
    df = df[df['medical_specialty'].isin(valid_categories)]
    df = df.reset_index(drop=True)

    return df