# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_class_distribution(df, column='medical_specialty', title=None, filename='class_distribution'):
    counts = df[column].value_counts()

    mean_val = counts.mean()
    median_val = counts.median()

    colors = ['#d9534f' if c < median_val else '#337ab7' for c in counts.values]

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(counts.index, counts.values, color=colors)

    ax.axvline(x=mean_val, color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.0f}')
    ax.axvline(x=median_val, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.0f}')

    for bar, count in zip(bars, counts.values):
        ax.text(count + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=8)

    blue_patch = mpatches.Patch(color='#337ab7', label='Above median')
    red_patch = mpatches.Patch(color='#d9534f', label='Below median (underrepresented)')
    ax.legend(handles=[blue_patch, red_patch,
                       mpatches.Patch(color='orange', label=f'Mean: {mean_val:.0f}'),
                       mpatches.Patch(color='green', label=f'Median: {median_val:.0f}')],
              loc='lower right')

    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Specialty')
    ax.set_title(title if title else f'Class Distribution — {column}\n'
                 f'Mean: {mean_val:.0f} | Median: {median_val:.0f} | '
                 f'Max/Min ratio: {counts.max() / counts.min():.1f}x imbalance')

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: data/{filename}.png")


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