import pandas as pd
from SmilesPE.pretokenizer import atomwise_tokenizer
from hdlib.space import Vector, Space
from hdlib.arithmetic import bundle, bind
from hdlib.model import MLModel
import random

df = pd.read_csv("tox21.csv")
df = df.dropna(subset=["NR-ER-LBD"]).reset_index(drop=True)

zero_set = list()
one_set = list()


for i, row in df.iterrows():
    val = row['NR-ER-LBD']
    smiles = row['smiles']

    print(f"{val}: {smiles}")

    if val == 1:
        one_set.append(smiles)
    else:
        zero_set.append(smiles)



zero_sample = random.sample(zero_set, 100)
one_sample = random.sample(one_set, 100)

shared_space = Space()

token_list = list()
for zero_vec, one_vec in zip(zero_sample, one_sample):
    tokens = atomwise_tokenizer(zero_vec)
    tokens2 = atomwise_tokenizer(one_vec)

    token_list.extend(tokens)
    token_list.extend(tokens2)


shared_space.bulk_insert(token_list)



def encode_sample_set(sample, shared_space):
    str_vec = dict()

    for hd_vec in sample:
        cur_tokens = atomwise_tokenizer(hd_vec)

        if len(cur_tokens) == 1:
            return shared_space.get(names=[cur_tokens[0]])[0]

        token_vec0 = shared_space.get(names=[cur_tokens[0]])[0]
        token_vec1 = shared_space.get(names=[cur_tokens[1]])[0]

        token_vec0.permute(rotate_by=0)
        token_vec1.permute(rotate_by=1)

        culmination = bind(token_vec0, token_vec1)

        for i in range(2, len(cur_tokens)):
            current_vec = shared_space.get(names=[cur_tokens[i]])[0]
            current_vec.permute(rotate_by=i)
            culmination = bind(culmination, current_vec)
        
        str_vec[hd_vec] = culmination

    mol_vecs = list(str_vec.values())

    class_vec = bundle(mol_vecs[0], mol_vecs[1])

    for i in range(2, len(mol_vecs)):
        current_vec = mol_vecs[i]
        class_vec = bundle(class_vec, current_vec)

    return class_vec


def encode_one_smiles(smiles, shared_space):
    tokens = atomwise_tokenizer(smiles)
    
    if len(tokens) == 1:
        return shared_space.get(names=[tokens[0]])[0]
    
    vec0 = shared_space.get(names=[tokens[0]])[0]
    vec1 = shared_space.get(names=[tokens[1]])[0]
    
    vec0.permute(rotate_by=0)
    vec1.permute(rotate_by=1)
    
    result = bind(vec0, vec1)
    
    for i in range(2, len(tokens)):
        v = shared_space.get(names=[tokens[i]])[0]
        v.permute(rotate_by=i)
        result = bind(result, v)
    
    return result


toxic_class_vec = encode_sample_set(zero_sample, shared_space)
nontoxic_class_vec = encode_sample_set(one_sample, shared_space)
