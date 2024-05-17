from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルの読み込み（1列目はデータラベルとして扱う）
df = pd.read_csv('/Users/*****/CrystalCondition.csv', index_col=0)

# データラベル（サンプル名）を保存
data_labels = df.iloc[:, 0]

# データの標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# PCAで2次元に削減
pca = PCA(2)
X_pca = pca.fit_transform(df_scaled)

# KMeansクラスタリング
n_clusters = 5  # クラスタの数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_pca)

# クラスタラベルの取得
y_kmeans = kmeans.predict(X_pca)

# クラスタリング結果のプロット
colors = ['orange', 'green', 'purple', 'red', 'brown']  # クラスタの数に応じて色を増やす
cluster_colors = [colors[label] for label in y_kmeans]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors, s=50)  # カスタム色を使用

# セントロイドのプロット
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering with 2D PCA')

# PNGファイルとして保存
plt.savefig('/Users/*****/kmeans_pca_result.png', dpi=600)

# 画面に表示
plt.show()

# クラスタリング結果をCSVに保存
df['Cluster'] = kmeans.labels_
df.to_csv('/Users/*****/Kmeans_result.csv')

# セントロイド（PCA変換後のスペースで）をCSVファイルに保存
centroids_pca = pd.DataFrame(centers, columns=['PC1', 'PC2'])
centroids_pca.to_csv('/Users/*****/centroids.csv', index=False)


# PCAで削減されたデータをDataFrameに変換し、サンプル名をインデックスとして設定
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = kmeans.labels_
pca_df.index = data_labels

# PCAで削減されたデータとサンプル名をCSVファイルに保存
pca_df.to_csv('/Users/*****/Kmeans_PCA.csv', index=False)