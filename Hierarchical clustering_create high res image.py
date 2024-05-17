import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('/hogehoge/CrystalCondition.csv')

# フィギュアのサイズを設定（インチ単位）
plt.figure(figsize=(15.748, 5.905))

# 1列目（データ名）を保持
labels = df.iloc[:, 0].values

# 数値データのみを含むDataFrameを作成（1列目を除外）
df_numeric = df.iloc[:, 1:]

# 階層的クラスタリングを実行
Z = linkage(df_numeric, method='ward')

# リンケージマトリックスのクラスタ間距離に対数変換を適用
Z_log = np.array(Z)
Z_log[:, 2] = np.log10(Z[:, 2] + 1.05)  # 対数変換（小さな値を足しておく）

# デンドログラムを描画
plt.figure(figsize=(40, 15))
dendrogram(Z_log, labels=labels, orientation='top')

# ラベルのフォントサイズを大きく設定
plt.tick_params(axis='x', which='major', labelsize=5)

# 枠線を消す
plt.box(False)
# ラベルが重ならないように回転
plt.xticks(rotation=90)  
# レイアウトの調整
plt.tight_layout()

# PNGファイルとして保存
plt.savefig('/hogehoge/dendrogram_log.png', dpi=600)

# 画面に表示（必要に応じて）
plt.show()