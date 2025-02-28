
import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
 
# Define the stock tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
 
# Define the date range
start_date = "2023-04-01"
end_date = "2024-02-10"
 
# Fetch historical stock prices from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
 
# Calculate daily returns
returns = data.pct_change().dropna()
 
# Compute correlation matrix
correlation_matrix = returns.corr()
 
# Create a graph from the correlation matrix
G = nx.Graph()
for i, stock1 in enumerate(tickers):
    for j, stock2 in enumerate(tickers):
        if i < j:
            G.add_edge(stock1, stock2, weight=correlation_matrix.loc[stock1, stock2])
 
# Spectral clustering
sc = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=42)
labels = sc.fit_predict(1 - correlation_matrix)
 
# Add labels to the graph
for i, stock in enumerate(tickers):
    G.nodes[stock]["label"] = labels[i]
 
# Calculate evaluation metrics
degree = dict(G.degree())
median_degree = np.median(list(degree.values()))
cut_edges = nx.cut_size(G, labels)
total_edges = G.number_of_edges()
 
# Separability
separability = cut_edges / total_edges
 
# Clustering coefficient
clustering_coefficient = nx.average_clustering(G)
 
# Fraction over median degree
fraction_over_median_degree = np.mean([degree[node] > median_degree for node in G.nodes])
 
# Conductance
conductance = cut_edges / min(sum(degree[node] for node in G.nodes if labels[i] == 0),
                               sum(degree[node] for node in G.nodes if labels[i] == 1))
 
# Normalized cut
normalized_cut = cut_edges / (sum(degree[node] for node in G.nodes if labels[i] == 0) +
                              sum(degree[node] for node in G.nodes if labels[i] == 1))
 
# Cut Ratio
cut_ratio = cut_edges / total_edges
 
# Print evaluation metrics
print(f"Separability: {separability:.4f}")
print(f"Clustering Coefficient: {clustering_coefficient:.4f}")
print(f"Fraction over Median Degree: {fraction_over_median_degree:.4f}")
print(f"Conductance: {conductance:.4f}")
print(f"Normalized Cut: {normalized_cut:.4f}")
print(f"Cut Ratio: {cut_ratio:.4f}")
 
