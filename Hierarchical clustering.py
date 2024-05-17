import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('/hogehoge/CrystalCondition.csv')

# 1列目（データ名）を保持
labels = df.iloc[:, 0].values

# 数値データのみを含むDataFrameを作成（1列目を除外）
df_numeric = df.iloc[:, 1:]

# 階層的クラスタリングを実行
Z = linkage(df_numeric, method='ward')

# デンドログラムの情報を辞書として取得（プロットは行わない）
dendro_dict = dendrogram(Z, labels=labels, no_plot=True)

# 実際にプロットを行う
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=labels, orientation='top', color_threshold=0.7 * max(Z[:, 2]))
plt.xticks(rotation=90)  # ラベルが重ならないように回転
plt.tight_layout()  # レイアウトの調整

# PNGファイルとして保存
plt.savefig('/hogehoge/dendrogram.png')

# 画面に表示
plt.show()

# 色の情報を含むラベルを表示
#print(dendro_dict)