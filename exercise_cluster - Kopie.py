import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = {
    'Age': [19, 21, 23, 31, 35, 40, 45, 46, 48, 52, 60, 63],
    'AnnualIncome': [15, 16, 17, 25, 30, 40, 50, 52, 55, 60, 80, 85],
    'SpendingScore': [39, 81, 6, 77, 40, 42, 65, 23, 73, 21, 88, 17]
}

df=pd.DataFrame(data)

print(df.head())

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'AnnualIncome', 'SpendingScore'])
print(df_scaled.head())

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

df['Cluster'] = kmeans.labels_

plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()