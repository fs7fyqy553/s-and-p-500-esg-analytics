from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

confusion_matrices = {
    'SimpleKMeans Seed=1': [
        # Columns from left to right are for Clusters 0-10 in order.
        [0, 0, 1, 0, 3, 8, 5, 0, 0, 10, 26],  # Healthcare
        [1, 5, 6, 9, 3, 3, 10, 1, 7, 4, 2],   # Consumer Cyclical
        [6, 8, 12, 1, 1, 0, 1, 1, 1, 1, 1],   # Consumer Defensive
        [1, 2, 2, 12, 0, 2, 17, 1, 9, 14, 1], # Technology
        [16, 3, 13, 4, 0, 0, 14, 0, 5, 2, 3], # Industrials
        [0, 0, 0, 26, 0, 0, 1, 0, 1, 0, 0],   # Real Estate
        [0, 0, 1, 0, 1, 5, 1, 0, 0, 3, 3],    # Communication Services
        [0, 0, 0, 0, 30, 25, 4, 0, 0, 3, 1],  # Financial Services
        [2, 16, 2, 0, 0, 0, 1, 2, 5, 0, 0],   # Utilities
        [1, 2, 1, 0, 0, 0, 0, 14, 2, 0, 0],   # Energy
        [2, 2, 1, 1, 0, 0, 0, 8, 5, 0, 0]     # Basic Materials
    ],
    'SimpleKMeans Seed=2': [
        [5, 4, 24, 0, 2, 0, 14, 0, 0, 0, 4],  # Healthcare
        [2, 3, 2, 2, 4, 8, 14, 2, 7, 0, 7],  # Consumer Cyclical
        [0, 1, 1, 3, 8, 10, 2, 6, 2, 0, 0],  # Consumer Defensive
        [2, 0, 1, 3, 3, 0, 24, 1, 10, 0, 17],  # Technology
        [0, 0, 2, 1, 15, 5, 10, 16, 7, 0, 4],  # Industrials
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 26],  # Real Estate
        [2, 1, 3, 0, 2, 0, 4, 0, 0, 0, 2],  # Communication Services
        [27, 31, 0, 0, 0, 0, 5, 0, 0, 0, 0],  # Financial Services
        [0, 0, 0, 13, 2, 8, 1, 2, 2, 0, 0],  # Utilities
        [0, 0, 0, 1, 0, 3, 0, 1, 2, 13, 0],  # Energy
        [0, 0, 0, 10, 0, 2, 0, 2, 4, 1, 0]  # Basic Materials
    ],
    'SimpleKMeans Seed=3': [
        [0, 0, 0, 28, 0, 4, 1, 2, 18, 0, 0],  # Healthcare
        [2, 0, 7, 4, 7, 7, 3, 1, 15, 0, 5],   # Consumer Cyclical
        [4, 4, 2, 2, 13, 0, 5, 0, 2, 0, 1],   # Consumer Defensive
        [3, 1, 17, 1, 2, 14, 0, 1, 19, 0, 3], # Technology
        [0, 14, 9, 2, 12, 2, 7, 0, 10, 0, 4], # Industrials
        [0, 0, 1, 0, 0, 25, 0, 0, 1, 0, 1],   # Real Estate
        [0, 0, 0, 4, 0, 2, 1, 2, 5, 0, 0],    # Communication Services
        [0, 0, 0, 7, 0, 0, 0, 41, 15, 0, 0],  # Financial Services
        [17, 2, 2, 0, 2, 0, 1, 0, 0, 0, 4],   # Utilities
        [1, 1, 1, 0, 2, 0, 1, 0, 0, 13, 1],   # Energy
        [7, 2, 1, 0, 0, 1, 0, 0, 0, 1, 7]     # Basic Materials
    ],
    'FarthestFirst Seed=1': [
        [0, 0, 9, 22, 0, 0, 1, 0, 14, 7, 0],  # Healthcare
        [0, 0, 13, 2, 4, 0, 0, 0, 27, 5, 0],  # Consumer Cyclical
        [8, 0, 1, 1, 0, 0, 1, 0, 20, 1, 1],   # Consumer Defensive
        [2, 0, 22, 1, 1, 0, 0, 0, 34, 1, 0],  # Technology
        [6, 0, 3, 2, 1, 0, 3, 0, 42, 0, 3],   # Industrials
        [0, 0, 25, 0, 1, 0, 0, 0, 2, 0, 0],   # Real Estate
        [0, 0, 4, 1, 0, 0, 1, 0, 6, 2, 0],    # Communication Services
        [0, 1, 4, 0, 0, 3, 0, 0, 13, 42, 0],  # Financial Services
        [8, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0],   # Utilities
        [3, 0, 0, 0, 0, 0, 0, 11, 5, 0, 1],   # Energy
        [4, 0, 0, 0, 5, 0, 0, 0, 7, 0, 3]     # Basic Materials
    ],
    'FarthestFirst Seed=2': [
        [24, 0, 1, 0, 0, 0, 19, 0, 9, 0, 0],  # Healthcare
        [26, 0, 0, 0, 4, 15, 2, 0, 4, 0, 0],  # Consumer Cyclical
        [4, 0, 1, 0, 0, 25, 1, 0, 1, 1, 0],   # Consumer Defensive
        [52, 0, 0, 0, 1, 5, 2, 0, 0, 0, 1],   # Technology
        [21, 0, 3, 0, 3, 26, 2, 0, 1, 3, 1],  # Industrials
        [27, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # Real Estate
        [8, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0],    # Communication Services
        [31, 0, 0, 1, 0, 0, 0, 3, 28, 0, 0],  # Financial Services
        [1, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0],   # Utilities
        [1, 7, 0, 0, 0, 5, 0, 0, 0, 0, 7],    # Energy
        [0, 0, 0, 0, 2, 8, 0, 0, 0, 1, 8]     # Basic Materials
    ],
    'FarthestFirst Seed=3': [
        [0, 0, 2, 4, 0, 10, 31, 0, 0, 6, 0],  # Healthcare
        [14, 0, 2, 0, 0, 16, 12, 3, 0, 4, 0],  # Consumer Cyclical
        [25, 0, 0, 1, 0, 2, 3, 0, 1, 1, 0],    # Consumer Defensive
        [9, 0, 1, 0, 0, 39, 11, 0, 0, 0, 1],   # Technology
        [30, 0, 0, 1, 0, 12, 12, 0, 5, 0, 0],  # Industrials
        [0, 0, 1, 0, 0, 25, 1, 1, 0, 0, 0],    # Real Estate
        [1, 0, 1, 1, 0, 2, 8, 0, 0, 1, 0],     # Communication Services
        [0, 1, 19, 0, 0, 1, 7, 0, 0, 35, 0],   # Financial Services
        [27, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],    # Utilities
        [7, 0, 0, 0, 7, 0, 0, 0, 0, 0, 6],     # Energy
        [9, 0, 0, 0, 0, 1, 0, 1, 3, 0, 5]      # Basic Materials
    ]
}

for clustering_method in confusion_matrices.keys():

    confusion_matrix = confusion_matrices[clustering_method]

    sector_names = ['Healthcare', 'Consumer Cyclical', 'Consumer Defensive', 'Technology', 'Industrials', 'Real Estate', 'Communication Services', 'Financial Services', 'Utilities', 'Energy', 'Basic Materials']

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=['\n'.join(label.split()) for label in sector_names],
        columns=[i for i in range(11)]
    )

    # === 
    # Confusion matrix
    # ===

    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='gray', cbar=False)
    plt.title(f'Confusion Matrix ({clustering_method})')
    plt.xlabel('Cluster')
    plt.ylabel('Sector')
    plt.yticks(fontsize=7)
    plt.show()

    # === 
    # Confusion matrix with only maximum frequency for each Sector
    # ===

    # Convert maxima to 100 and others to 0
    highlighted_df = confusion_matrix_df.apply(
        lambda row: (row == row.max()).astype(int) * 100, axis=1
    )

    sns.heatmap(highlighted_df, fmt='.0f', cmap='gray_r', cbar=False, linewidths=1, linecolor='black')
    plt.title(f'Primary Cluster For Each Sector ({clustering_method})')
    plt.xlabel('Cluster')
    plt.ylabel('Sector')
    plt.yticks(fontsize=7)
    plt.show()

    # === 
    # Confusion matrix with only maximum frequency for each Sector (with percentages shown)
    # ===

    # Calculating the percentages for each row
    highlighted_df = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0) * 100

    # Retaining only the maximum percentage in each row, setting others to 0
    highlighted_df = highlighted_df.apply(
        lambda row: row.where(row == row.max(), other=0), axis=1
    )

    # Creating annotation data where zeros are replaced with empty strings
    annotations = highlighted_df.map(lambda x: f"{x:.1f}" if x > 0 else "")

    sns.heatmap(
        highlighted_df, 
        annot=annotations,
        fmt='',  # The annotations are already formatted as strings
        cmap='gray_r', 
        cbar=False, 
        linewidths=1, 
        linecolor='black'
    )
    plt.title(f'Primary Cluster Percentage For Each Sector ({clustering_method})')
    plt.xlabel('Cluster')
    plt.ylabel('Sector')
    plt.yticks(fontsize=7)
    plt.show()

    # === 
    # Calculating ARI values 
    # ===

    # The below are necessary arguments for sklearn.metrics.adjusted_rand_score to calculate ARI for this clustering. Between the two lists, a common index is for a single company. The first list will have the company's sector (the class) and the second will have its cluster.
    true_labels = []
    predicted_labels = []

    for sector_index, row in enumerate(confusion_matrix):
        for cluster_index, count in enumerate(row):
            true_labels.extend([sector_index] * count)
            predicted_labels.extend([cluster_index] * count)

    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(ari)
