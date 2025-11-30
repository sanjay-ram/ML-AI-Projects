import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = {'AnnualIncome': [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
        'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 5, 82, 56],
        'Age': [20, 22, 25, 24, 35, 40, 30, 21, 50, 31]}

df = pd.DataFrame(data)
print(df.head())

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print(df_scaled.head())

kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['Kmeans_Cluster'] = kmeans.fit_predict(df_scaled)

plt.scatter(df_scaled['AnnualIncome'], df_scaled['SpendingScore'], c=df_scaled['Kmeans_Cluster'], cmap='virdis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print(df_pca.head())
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_scaled['Kmeans_Cluster'], cmap='virdis')
plt.title("PCA KMeans Clustering")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)
df_tsne = pd.DataFrame(df_tsne, columns=['TSNE_1', 'TSNE_2'])
print(df_tsne.head())
plt.scatter(df_tsne['TSNE_1'], df_tsne['TSNE_2'], c=df_scaled['Kmeans_Cluster'], cmap='virdis')
plt.title("TSNE KMeans Clustering")
plt.xlabel("TSNE_1")
plt.ylabel("TSNE_2")
plt.show()