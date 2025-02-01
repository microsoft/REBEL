import pandas as pd
import matplotlib.pyplot as plt

# Data from one run on the dataset, for validation.
data = {
    'Experiment': [
        'VDB + Cohere rerank',
        'VDB + Corrected Our Method',
        'VDB + HyDE',
        'VDB + HyDE + Cohere Rerank',
        'VDB + HyDE + Corrected Our Method',
        'VDB + HyDE + LLM Rerank',
        'VDB + HyDE + Static Rerank',
        'VDB + LLM Rerank',
        'VDB + MMR',
        'VDB + MMR + HyDE',
        'VDB + Static Rerank',
        'Vanilla'
    ],
    'RetrievalPrecision': [0.8754, 0.9418, 0.8910, 0.9034, 0.9487, 0.9545, 0.9190, 0.9276, 0.8442, 0.8723, 0.8847, 0.8474],
    'AnswerSimilarity': [4.1028, 4.3551, 4.2897, 4.0935, 4.2991, 4.1215, 4.2897, 4.0748, 4.1869, 4.3178, 4.3458, 4.2056]
}

def plot_results():
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot RetrievalPrecision
    ax1.barh(df['Experiment'], df['RetrievalPrecision'], color='skyblue')
    ax1.set_title('Retrieval Precision Scores', pad=20)
    ax1.set_xlabel('Score')
    ax1.grid(True, axis='x')
    
    # Plot AnswerSimilarity
    ax2.barh(df['Experiment'], df['AnswerSimilarity'], color='lightgreen')
    ax2.set_title('Answer Similarity Scores', pad=20)
    ax2.set_xlabel('Score')
    ax2.grid(True, axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'benchmark_results.png'")

if __name__ == "__main__":
    plot_results() 