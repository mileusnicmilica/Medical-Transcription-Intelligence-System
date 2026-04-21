# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_class_distribution(df, column='medical_specialty'):
    """
    Creates a bar plot of medical specialties to visualize imbalance.
    Professional tip: Use this before and after applying SMOTE.
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(y=df[column], order=df[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Specialty')
    plt.tight_layout()
    plt.show()


def plot_wordcloud(df, specialty_name):
    """
    Generates a word cloud for a specific medical specialty.
    Useful for explaining what the model 'sees' in the text.
    """
    text = " ".join(df[df['medical_specialty'] == specialty_name]['cleaned_transcription'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {specialty_name}')
    plt.axis('off')
    plt.show()