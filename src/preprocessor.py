import spacy
from tqdm import tqdm
import pandas as pd

class MedicalPreprocessor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes the SpaCy NLP pipeline.
        Defaulting to the small English model for efficiency.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # If the model isn't downloaded
            raise OSError(f"Model {model_name} not found. Run: python -m spacy download {model_name}")

    def clean_text(self, text):
        """
        Performs advanced preprocessing using SpaCy:
        1. Tokenization and Lemmatization (converting words to their base form).
        2. Removal of stop words, punctuation, and special characters.
        3. Lowercasing.

        Args:
            text (str): Raw medical transcription text.

        Returns:
            str: Preprocessed and lemmatized string.
        """
        if not isinstance(text, str):
            return ""

        # Process the text with the SpaCy pipeline
        doc = self.nlp(text.lower(), disable=['ner', 'parser']) # should be faster than doc = self.nlp(text.lower())?

        # Extract lemmas for non-stop words and non-punctuation tokens
        # We also filter out tokens that are mostly numbers (common in medical reports)
        clean_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num
        ]

        return " ".join(clean_tokens)



    def preprocess_dataframe(self, df, column_name='transcription'):
        print(f"Starting SpaCy preprocessing...")
        # adding progress bar
        tqdm.pandas()
        df[f'cleaned_{column_name}'] = df[column_name].progress_apply(self.clean_text)
        return df