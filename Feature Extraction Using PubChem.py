import pubchempy as pcp
import pandas as pd

# CSVファイルからCAS番号を読み込む
cas_df = pd.read_csv('/Users/******/*****.csv') 
cas_list = cas_df['CAS'].tolist()

data = []  # 結果を保存するためのリスト

for cas in cas_list:
    try:
        compounds = pcp.get_compounds(cas, 'name')  # CAS番号から化合物を取得
        if compounds:
            compound = compounds[0]  # 最初の化合物を使用
            temp = {
                'CAS': cas,
                'MolecularFormula': compound.molecular_formula,
                'MolecularWeight': compound.molecular_weight,
                'XLogP': compound.xlogp,
                'TPSA': compound.tpsa,
                'RotatableBondCount': compound.rotatable_bond_count,
                'HBondDonorCount': compound.h_bond_donor_count,
                'HBondAcceptorCount': compound.h_bond_acceptor_count,
                'HeavyAtomCount': compound.heavy_atom_count,
                'Complexity': compound.complexity,
            }
            data.append(temp)
    except Exception as e:
        print(f"Error retrieving data for {cas}: {e}")

# 結果をDataFrameに変換
df = pd.DataFrame(data)

# DataFrameをCSVファイルに出力
output_path = '/hogehoge/output.csv' 
df.to_csv(output_path, index=False) 