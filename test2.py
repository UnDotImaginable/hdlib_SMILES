import pandas as pd
import numpy as np

def zeroes_to_ones(column):
    df = pd.read_csv("smiles_code/tox21.csv")
    df = df.dropna(subset=[column]).reset_index(drop=True)

    z_count = 0
    o_count = 0

    for i, row in df.iterrows():
        val = row[column]
        smiles = row['smiles']

        if (val == 0):
            z_count += 1
        else:
            o_count += 1


    print("\n" + str(column))
    print(str(z_count))
    print(o_count)

zeroes_to_ones("NR-AR")
zeroes_to_ones("NR-AR-LBD")
zeroes_to_ones("NR-AhR")
zeroes_to_ones("NR-Aromatase")
zeroes_to_ones("NR-ER")
zeroes_to_ones("NR-ER-LBD")
zeroes_to_ones("NR-PPAR-gamma")
zeroes_to_ones("SR-ARE")
zeroes_to_ones("SR-ATAD5")
zeroes_to_ones("SR-HSE")
zeroes_to_ones("SR-MMP")
zeroes_to_ones("SR-p53")

