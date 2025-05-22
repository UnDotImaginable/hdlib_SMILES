import codecs
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from hdlib.space import Vector
from hdlib.space import Space
from hdlib.arithmetic import bind, bundle
from collections import Counter
import pandas as pd
import math


instance_proportion = dict()
filtered_proportion = dict()

df = pd.read_csv("tox21.csv")

for index, row in df.iterrows():
    zero_count = 0
    one_count = 0
    smile_str = df.iloc[index, 13]

    for i in range(0, 12):
        value = df.iloc[index, i]

        if pd.isnull(value):
            continue
        else:
            if (value == 0):
                zero_count += 1
            if (value == 1):
                one_count += 1

    instance_proportion[smile_str] = [zero_count / (zero_count + one_count), one_count / (zero_count + one_count)]
    zero_count = 0
    one_count = 0

for key, value in instance_proportion.items():
    ratio = min(value) / max(value)

    if (ratio >= 0.35):
        filtered_proportion[key] = ratio



shared_space = Space()

pieces = list()
culmination_vec_list = dict()



for key in filtered_proportion.keys():
    tokens = atomwise_tokenizer(key)
    pieces.extend(tokens)


shared_space.bulk_insert(pieces)




for key in filtered_proportion.keys():
    tokens = list(atomwise_tokenizer(key))

    if (len(tokens) < 2):
        culmination_vec_list[key] = shared_space.get(names=[tokens[0]])[0]
    else:
        token_vec0 = shared_space.get(names=[tokens[0]])[0]
        token_vec1 = shared_space.get(names=[tokens[1]])[0]

        token_vec0.permute(rotate_by=0)
        token_vec1.permute(rotate_by=1)

        culmination = bind(token_vec0, token_vec1)

        for i in range(2, len(tokens)):
            current_vec = shared_space.get(names=[tokens[i]])[0]
            current_vec.permute(rotate_by=i)
            culmination = bind(culmination, current_vec)

        culmination_vec_list[key] = culmination



vectors_only = list(culmination_vec_list.values())

class_vec = bundle(vectors_only[0], vectors_only[1])

for i in range(2, len(vectors_only)):
    current_vec = vectors_only[i]
    class_vec = bundle(class_vec, current_vec)

print(class_vec)


def encode_single_smiles(smi, vec_space):
    tokens = atomwise_tokenizer(smi)

    if len(tokens) == 1:
        return vec_space.get(names=[tokens[0]])[0]


    token_vec0 = vec_space.get(names=[tokens[0]])[0]
    token_vec1 = vec_space.get(names=[tokens[1]])[0]

    token_vec0.permute(rotate_by=0)
    token_vec1.permute(rotate_by=1)

    culmination = bind(token_vec0, token_vec1)

    for i in range(2, len(tokens)):
        current_vec = vec_space.get(names=[tokens[i]])[0]
        current_vec.permute(rotate_by=i)
        culmination = bind(culmination, current_vec)

    return culmination



toxic_ex = "C#N"
nontoxic_ex = "O"

toxic_ex_vec = encode_single_smiles(toxic_ex, shared_space)

nontoxic_ex_vec = encode_single_smiles(nontoxic_ex, shared_space)



print(f"{toxic_ex_vec.dist(class_vec, method='cosine')}\n")
print(f"{nontoxic_ex_vec.dist(class_vec, method='cosine')}")

if toxic_ex_vec.dist(class_vec, method='cosine') < nontoxic_ex_vec.dist(class_vec, method='cosine'):
    print("The toxic molecule is closer to \"s\" than the non-toxic one")
else:
    print("The non-toxic molecule is closer to \"s\" than the toxic one")



    