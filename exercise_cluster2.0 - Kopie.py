import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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

dbscan = DBSCAN(eps=0.7, min_samples=3)
dbscan.fit(df_scaled)

df['Cluster'] = dbscan.labels_

plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.xlabel('AnnualIncome')
plt.ylabel('SpendingScore')
plt.show()