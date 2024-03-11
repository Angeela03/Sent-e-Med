import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np
from scipy.spatial.distance import cdist

# pickle_file = "/medbert_new_task/MIMIC_new_task.types"

# data_path = "/Med-BERT/data/"
icd_mapping = pd.read_csv(os.path.join(data_path, "icd_mapping.csv"))
icd_mapping["icd_code"] = icd_mapping["icd_code"].str.zfill(3)


def add_string(x):
    if len(x)>3:
        res = x[ :3] + "." + x[3: ]
    else:
        res = x
    return res


final_merge_copy = icd_mapping.copy()
final_merge_copy["diag"] = final_merge_copy["icd_code"].apply(add_string)
final_merge_copy["icd_version"] = "ICD" + final_merge_copy["icd_version"] .astype(str)
final_merge_copy["diag"] = final_merge_copy["icd_code"].apply(add_string)
final_merge_copy["diag"] = final_merge_copy["icd_version"] + "_"+ final_merge_copy["diag"]

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded")
print("Getting embeddings")
sentence_embeddings = model.encode(list(final_merge_copy["long_title"].values))
res = dict(zip(list(final_merge_copy["diag"].values), list(sentence_embeddings)))
print("Enbeddings obtained")
print("Getting only the necessary ones")
with open(os.path.join(data_path, pickle_file), 'rb') as fp:
    types = pickle.load(fp)
print(types)


arr_all = []
count = 0
for i in types.keys():
    if i == "empty_pad":
        arr = np.zeros(384)
    else:
        try:
            arr = res[i]
        except:
            if i=="ICD10_nan":
                print(i)
                arr = np.zeros(384)
            else:   
                try:
                    find_dot = i.find(".")
                    i_parent = i[:find_dot]
                    arr=res[i_parent]
                except:
                    count+=1
                    arr = np.zeros(384)
                    pass
    arr_all.append(arr)

# dict_final = dict(zip(list(types.values()), arr_all))


np.save('token_embeddings.npy', arr_all, allow_pickle=True)
print("Process completed")




