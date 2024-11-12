import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load image paths and labels from a CSV file
def load_image_data(csv_file, image_folder):
    data = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, f"{row['id']}.png")
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
            labels.append(row['label'])
    return images, labels

# Compute color histogram for each image
def compute_color_histogram(images, bins=(8, 8, 8)):
    histograms = []
    for image in images:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return np.array(histograms)

# Perform clustering
def perform_clustering(data, n_clusters=10):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering_model.fit_predict(data)
    return clusters

# Convert labels to numeric
def encode_labels(labels):
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    return [label_mapping[label] for label in labels]

# Visualize clusters using t-SNE
def visualize_tsne(data, clusters, labels=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='tab10', marker='o')
    plt.colorbar(scatter, ticks=range(10), label='Cluster')
    
    if labels:
        numeric_labels = encode_labels(labels)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=numeric_labels, cmap='tab10', alpha=0.2, marker='x')
        plt.title("t-SNE Visualization with True Labels Overlay")
    else:
        plt.title("t-SNE Visualization of Clusters")
    plt.show()

# Generate and plot a dendrogram for hierarchical clustering
def plot_dendrogram(data):
    Z = linkage(data, method='ward')
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title("Dendrogram of Image Clusters")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

    # Calculate clustering metrics
def calculate_clustering_metrics(data, clusters, labels):
    silhouette = silhouette_score(data, clusters)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    
    print("Silhouette Score:", silhouette)
    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)

# Main function to load data, compute histograms, cluster, and visualize
def main(csv_file, image_folder, cluster_values=[10, 15, 20]):
    images, labels = load_image_data(csv_file, image_folder)
    histograms = compute_color_histogram(images)
    
    scaler = StandardScaler()
    histograms = scaler.fit_transform(histograms)

    # Iterate over different cluster values
    for n_clusters in cluster_values:
        print(f"\nClustering with {n_clusters} clusters:")
        
        # Perform clustering
        clusters = perform_clustering(histograms, n_clusters=n_clusters)
        
        # Visualize t-SNE for the current clustering result
        visualize_tsne(histograms, clusters, labels)
        
        # Plot dendrogram for the current clustering result
        plot_dendrogram(histograms)

        # Calculate and print clustering metrics
        calculate_clustering_metrics(histograms, clusters, encode_labels(labels))

# Set paths and run the clustering for different cluster counts
csv_file = 'training_CIFAR.csv'
image_folder = 'train/'
main(csv_file, image_folder, cluster_values=[10, 15, 20])
