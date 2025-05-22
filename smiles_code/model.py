import pandas as pd
from hdlib.space import Vector, Space
from hdlib.arithmetic import bundle, permute
from hdlib.model import MLModel
import random
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from sklearn.metrics import matthews_corrcoef


memory = Space() # ITEM MEMORY

df = pd.read_csv("smiles_code/tox21.csv")
df = df.dropna(subset=["NR-ER-LBD"]).reset_index(drop=True)

all_entries = list()
to_insert = list()

for i, row in df.iterrows():
    val = row['NR-ER-LBD']
    smiles = row['smiles']

    all_entries.append((val, smiles))

    tokens = atomwise_tokenizer(smiles)
    to_insert.extend(tokens)

memory.bulk_insert(to_insert)

random.shuffle(all_entries)
split_index = int(0.8 * len(all_entries))
sample_80 = all_entries[:split_index]
sample_20 = all_entries[split_index:]

zero_vecs = [all_entries[i][1] for i in range(0, len(sample_80)) if all_entries[i][0] == 0]
one_vecs = [all_entries[i][1] for i in range(0, len(sample_80)) if all_entries[i][0] == 1]


def encode_sample(sample, shared_space): # Input: sample of SMILES strings, item memory
    str_vec = dict() # Initialize empty list of key-value pairs

    for hd_vec in sample: # For every SMILES string inside sample...
        cur_tokens = atomwise_tokenizer(hd_vec) # Tokenize string into chemically meaningful tokens

        if len(cur_tokens) == 1: # If length of cur_tokens (a list of tokens) is 1...
            """
                The shared_space parameter will take in a shared_space variable from earlier in the code that we established
                This space contains a unique vector for every single unique token in every single SMILES string in our chosen column (which represents an assay)
                If this SMILES string can only be broken down into ONE unique token, then we simply retrieve the corresponding HD vector of that token from the shared_space
                The use of "[0]" is necessary as both cur_tokens and the result of the .get() function are LISTS. We do not want to return a list but rather an element INSIDE
                    the list. "[0]" retrieves the 0-th (or 1st) index in a list in Python
            """
            return shared_space.get(names=[cur_tokens[0]])[0] # 

        # Otherwise...
        token_vec0 = shared_space.get(names=[cur_tokens[0]])[0] # Retrieve HD vector of 1st token in cur_tokens
        token_vec1 = shared_space.get(names=[cur_tokens[1]])[0] # Retrieve HD vector of 2nd token in cur_tokens

        token_vec0.permute(rotate_by=0) # Rotate the 1st HD vector by 0 spaces
        token_vec1.permute(rotate_by=1) # Rotate the 2nd HD vector by 1 space

        culmination = bundle(token_vec0, token_vec1) # Bundle the permuted vectors together into a "culmination" vector 

        for i in range(2, len(cur_tokens)): # For every token from the 3rd token (at index 2) to the end of the list...
            current_vec = shared_space.get(names=[cur_tokens[i]])[0] # Retrieve the HD vector corresponding to the token from the shared space.
            current_vec.permute(rotate_by=i) # Permute it by its placement (i.e., index) in the SMILES string to encode its position into the vector
            culmination = bundle(culmination, current_vec) # Bundle the permuted vector with the culmination vector

        str_vec[hd_vec] = culmination # Once the loop terminates, enter the current SMILES string (key) and its HD vector (value) into str_vec.

    mol_vecs = list(str_vec.values()) # Retrieve only the vectors from the list of key-value pairs

    class_vec = bundle(mol_vecs[0], mol_vecs[1]) # Bundle the 1st and 2nd values from that list

    # Repeat the same operation as above for this list of HD vectors 
    for i in range(2, len(mol_vecs)):
        current_vec = mol_vecs[i]
        class_vec = bundle(class_vec, current_vec)

    # The final class vector is returned by the function at the end.
    return class_vec



def encode_smi(smiles, shared_space): # Input: a single SMILES string, a pre-established shared space
    tokens = atomwise_tokenizer(smiles) # Tokenize the SMILES string into chemically meaningful tokens

    if len(tokens) == 1: # If the tokenization yields just one single token...
        return shared_space.get(names=[tokens[0]])[0] # Return that token's corresponding HD vector from the shared space

    vec0 = shared_space.get(names=[tokens[0]])[0] # Retrieve corresponding HD vector of the 1st token
    vec1 = shared_space.get(names=[tokens[1]])[0] # Retrieve corresponding HD vector of the 2nd token

    vec0.permute(rotate_by=0) # Permute HD vector of 1st element by 0 spaces
    vec1.permute(rotate_by=1) # Permute HD vector of 2nd element by 1 space

    result = bundle(vec0, vec1) # Combine the two vectors together

    for i in range(2, len(tokens)): # For every token in the list of tokens from the 3rd token to the end of the list...
        v = shared_space.get(names=[tokens[i]])[0] # Retrieve the token's corresponding HD vector
        v.permute(rotate_by=i) # Permute the vector by the number of its index in the list of tokens
        result = bundle(result, v) # Bind the vector with the combined vector from earlier

    return result # Return this vector as the result of this function


K = 5 # Number of folds 
"""
    A KFold object with the following parameters:
        1. Number of splits = k (or 5)
        2. "shuffle" is True
            i. Data will be shuffled before being split into batches
        3. "random_state" is 40
            i. Sets the seed of the random generator to allow shuffling of elements the same way each time
"""
kf = KFold(n_splits=K, shuffle=True, random_state=40)


"""
    The "enumerate" function in Python takes an Iterable item (list, dictionary, set) and converts it into a list of tuples where each tuple
        contains the index and the corresponding element
    
    In our case, every element in "enumerate(kf.split(all_entries))" is as follows:
        - index, (list of the indices of all elements that constitute our training set, list of the indices of all elements that constitute our testing set)
"""
for fold_idx, (training_indices, testing_indices) in enumerate(kf.split(all_entries)):
    real_all, pred_all = [], [] # Where we'll store the real vs. predicted labels

    """
        "all_entries" is a list of tuples where each entry is as follows:
            - (label value, SMILES string)
        
        Thus, here, we're compiling two lists for training and testing, respectively, that uses the indices found in "training_indices" and "testing_indices"
            to fetch the correct elements.
    """
    train_data = [all_entries[i] for i in training_indices]
    test_data = [all_entries[i] for i in testing_indices]

    """
        Given the structure of "all_entries", we need to iterate over both elements in the tuples (label, string)

        Here, we're compiling two lists from the training data: one consisting of SMILES strings whose class label is 0 and
            one consisting of SMILES strings whose class label is 1
    """
    zero_vecs = [smi for lbl, smi in train_data if lbl == 0]
    one_vecs = [smi for lbl, smi in train_data if lbl == 1]


    zero_cv = encode_sample(zero_vecs, memory) # Encode the list of "zero" vectors using our item memory and the function "encode_sample()"
    one_cv = encode_sample(one_vecs, memory) # Encode the list of "one" vectors using our item memory and the function "encode_sample()"

    # This is where the error mitigation section starts

    MAX_ITERS = 20  # max iterations to prevent infinite loops 
    prev_error_rate = float('inf') # Initialize an empty float variable of positive infinity (for debugging purposes, this setup allowed the least number of issues)

    # For every iteration up to MAX_ITERS (20)...
    for iteration in range(MAX_ITERS):
        
        # Counter variables to count the number of misclassified SMILES strings and the total number of SMILES strings processed
        misclassified = 0
        total = 0
        
        # For every label, SMILES string in "train_data" (from earlier)...
        for lbl, smiles in train_data:
            total += 1 # Increment "total" by 1 (we've processed this current string)

            vec_rep = encode_smi(smiles, memory) # Encode current string
            dist0 = vec_rep.dist(zero_cv, method="cosine") # Calculate its cosine distance from the zero label class vector 
            dist1 = vec_rep.dist(one_cv, method="cosine") # Calculate its cosine distance from the one label class vector

            pred = 0 if dist0 < dist1 else 1 # "pred" = 0 if distance from zero_cv is less distance from one_cv. Otherwise, its 1.

            if (pred != lbl): # If we mis-predicted this string...
                misclassified += 1 # Increment "misclassified" by 1 (we've made a mistake)


                if (pred == 0): # Our prediction is 0, label is 1

                    """
                    With this error mitigation step, our goal is very simple:
                        1. Increase the presence of the current string in its true class vector (0 or 1) via element-wise addition
                        2. Decrease the presence of the current string in its pred class vector (0 or 1) via element-wise subtraction
                    
                    We achieve this by: 
                        1. Converting the string's encoded Vector() object into a numpy array and adding it, element-wise, to the class vector of its true class
                            i. We then convert this modified numpy array back into a Vector() object and re-assign it back to its variable (essentially updating it)
                        2. We repeat the previous step but with the class vector of the predicted class and, instead of adding, we subtract.
                    """
                    upwards_cv = np.add(one_cv.vector, vec_rep.vector)
                    one_cv = Vector(vector=upwards_cv)

                    # Goal 2: Decrease presence in the false class vector
                    downwards_cv = np.subtract(zero_cv.vector, vec_rep.vector)
                    zero_cv = Vector(vector=downwards_cv)

                else:

                    # Goal 1: Increase presennce in the true class vector
                    plus_cv = np.add(zero_cv.vector, vec_rep.vector)
                    zero_cv = Vector(vector=plus_cv)

                    # Goal 2: Decrease presence in the false class vector
                    minus_cv = np.subtract(one_cv.vector, vec_rep.vector)
                    one_cv = Vector(vector=minus_cv)


        # Compute error rate (# of misclassified strings divided by the total number of strings)
        error_rate = misclassified / total
        print(f"Iteration {iteration+1}: Error rate = {error_rate:.4f}")

        # The error mitigation stops if the error rate of the previous iteration is less than the error rate of this one
        # Otherwise, this process can go on for a VERY long time.
        if error_rate > prev_error_rate:
            print("Stopping: Error rate increased.")
            break

        prev_error_rate = error_rate # Update the previous error rate to this error rate (the error rate of this iteration)


# -----------------------------------------------------------------------------------------

    # This is where the actual prediction happens (once error mitigation has provided as much benefit as possible)

    # For every label, SMILES string in test_data...
    for lbl, smiles in test_data:
        vec_rep = encode_smi(smiles, memory) # Encode the SMILES string into a HD vector
        dist0 = vec_rep.dist(zero_cv, method="cosine") # Calculate vector's cosine distance from the zero class vector
        dist1 = vec_rep.dist(one_cv, method="cosine") # Calculate vector's cosine distance from the one class vector

        pred = 0 if dist0 < dist1 else 1 # Make the prediction based on the calculated distances

        real_all.append(lbl) # Append the true label to "real_all"
        pred_all.append(pred) # Append the predicted label to "pred_all"
        # Since both of these lists were initialized up above as empty, they'll both end up having the same number of elements within.


    """ ---------- METRICS ---------- """

    print(f"\t\tIteration {fold_idx + 1}")
    labels = [0, 1]

    """
    This is a built-in method from the famous Python library "scikit-learn" which prints out the number of true labels and predicted labels for each fold
    We've just formatted it to make it easier to read. It looks something like this:

                    Pred 0  Pred 1
        True 0    1167     157
        True 1      60       7

    """
    cm = metrics.confusion_matrix(real_all, pred_all, labels=labels) 
    df_cm = pd.DataFrame(cm, index=[f"True {l}" for l in labels],
                            columns=[f"Pred {l}" for l in labels])
    print("Confusion Matrix (cross-validated):")
    print(df_cm)

    # This is a built-in method from the famous Python library "scikit-learn" which prints out metrics such as precision, recall, accuracy, and F1-score
    print("\nClassification Report:")
    print(metrics.classification_report(real_all, pred_all, labels=labels, digits=4))

    # This is another method from "scikit-learn" which calculates and prints out a metric called the Matthews Correlation Coefficient (also known as the phi coefficient)
    # This metric is often used in machine learning to evaluate the quality of classification, especially when dealing with imbalanced datasets. 
    # It measures the correlation between the true and predicted labels, considering all four elements of the confusion matrix
    # MCC ranges from -1 to +1, with +1 indicating perfect prediction, 0 indicating random prediction, and -1 indicating a completely opposite prediction
    mcc = matthews_corrcoef(real_all, pred_all)
    print(f"MCC: {mcc}\n")




