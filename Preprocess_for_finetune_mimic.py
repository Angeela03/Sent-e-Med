import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import sys

data_path = "/Med-BERT/data/"
data_path_2 = "/medbert_new_task/"

with open(os.path.join(data_path_2, 'MIMIC_new_task.bencs.train'), 'rb') as fp:
    train = pickle.load(fp)

with open(os.path.join(data_path_2, 'MIMIC_new_task.bencs.test'), 'rb') as fp:
    test = pickle.load(fp)

with open(os.path.join(data_path_2, 'MIMIC_new_task.bencs.valid'), 'rb') as fp:
    valid = pickle.load(fp)


def find_labels(list_req, read_MIMIC_bert):
    list_np = np.array(list_req)
    data_ = read_MIMIC_bert[read_MIMIC_bert["patient_id"].isin(list_np[:, 0])]
    unique_patient_label = data_[["patient_id", "label"]].drop_duplicates()
    label_1 = unique_patient_label[unique_patient_label["label"] == 1]
    label_0 = unique_patient_label[unique_patient_label["label"] == 0]
    print(len(label_1), len(label_0))
    return data_


def data_preprocessing(n_data, type, output_file):
    n_data.columns = ['patient_sk', 'admit_dt_tm', 'discharge_dt_tm', 'diagnosis', 'poa', 'diagnosis_priority',
                        'label', 'third_party_ind']

    ##### Data pre-processing
    print('Start Data Preprocessing !!!')
    count = 0
    pts_sls = []

    for Pt, group in n_data.groupby('patient_sk'):
        full_seq = []
        v_seg = []
        pt_discdt = []
        pt_addt = []
        v = 0
        # print(Pt)
        label = group["label"].values[0]

        for Time, subgroup in group.sort_values(['discharge_dt_tm'],
                                                ascending=True).groupby('discharge_dt_tm',
                                                                        sort=False):
            v = v + 1
            diag_l = np.array(subgroup['diagnosis'].drop_duplicates()).tolist()

            if len(diag_l) > 0:
                diag_lm = []

                for code in diag_l:
                    if code in types:
                        diag_lm.append(types[code])
                    else:
                        types[code] = max(types.values()) + 1
                        diag_lm.append(types[code])

                    v_seg.append(v)

                full_seq.extend(diag_lm)
            pt_discdt.append(Time)
            pt_addt.append(min(np.array(subgroup['admit_dt_tm'].drop_duplicates()).tolist()))

        pts_sls.append([Pt, label, full_seq, v_seg])

        count = count + 1

        if count % 1000 == 0: print('processed %d pts' % count)

    print('dumping %d pts' % count)

    bertencsfile = data_path + "MIMIC_new_task_finetune_" + output_file + '.bencs.' + type
    pickle.dump(pts_sls, open(bertencsfile, 'a+b'), protocol=2)
    pickle.dump(types, open(data_path + "MIMIC_new_task_finetune_" + output_file + '.types', 'wb'), protocol=2)

        # if count % 5000 == 0:
        #     split_fn(pts_ls,pts_sls,outFile)
        #     pts_ls=[]
        #     pts_sls=[]


if __name__=="__main__":
    req_file = sys.argv[1]
    output_file = sys.argv[2]
    read_MIMIC_bert = pd.read_csv(os.path.join(data_path, req_file))

    # Reformat EHR data
    def add_string(x):
        if len(x) > 3:
            res = x[:3] + "." + x[3:]
        else:
            res = x
        return res

    final_merge_copy = read_MIMIC_bert.copy()
    final_merge_copy["diag"] = final_merge_copy["icd_code"].apply(add_string)
    final_merge_copy["icd_version"] = "ICD" + final_merge_copy["icd_version"].astype(str)
    final_merge_copy["diag"] = final_merge_copy["icd_version"] + "_" + final_merge_copy["diag"]
    final_merge_copy['dischtime'] = pd.to_datetime(final_merge_copy['dischtime'], format='%Y-%m-%d').dt.date
    final_merge_copy['admittime'] = pd.to_datetime(final_merge_copy['admittime'], format='%Y-%m-%d').dt.date
    final_merge_copy["poa"] = 1

    final_merge_copy = final_merge_copy[["subject_id", "admittime", "dischtime", "diag", "poa", "seq_num", "label"]]
    final_merge_copy.columns = ["patient_id", "vadate", "vddate", "diag", "poa", "diagnosis_priority", "label"]
    final_merge_copy["third_part_ind"] = 0

    # Remove patients that have less than 3 diagnosis
    groupby_patient_id = final_merge_copy.groupby("patient_id").count()
    filter_less_than_three = groupby_patient_id[groupby_patient_id["diag"] > 3]
    less_than_three_df = final_merge_copy[final_merge_copy["patient_id"].isin(filter_less_than_three.index)]
    # print(less_than_three_df)

    print(len(np.unique(less_than_three_df[less_than_three_df["label"] == 1]["patient_id"])))
    print(len(np.unique(less_than_three_df[less_than_three_df["label"] == 0]["patient_id"])))

    train_data = find_labels(train, less_than_three_df)
    test_data = find_labels(test, less_than_three_df)
    validation_data = find_labels(valid, less_than_three_df)


    train_data = find_labels(train, less_than_three_df)
    test_data = find_labels(test, less_than_three_df)
    validation_data = find_labels(valid, less_than_three_df)

    # Prepare pretrain for fine tune
    with open(os.path.join(data_path_2, "MIMIC_new_task.types"), 'rb') as t2:
        types = pickle.load(t2)

    # if typeFile == 'NA':
    #     types = {'empty_pad': 0}
    #     types_3_digit = {'empty_pad': 0}
    # else:
    #     with open(typeFile + ".types", 'rb') as t2:
    #         types = pickle.load(t2)
    #     with open(typeFile + "3_digit.types", 'rb') as t3:
    #         types_3_digit = pickle.load(t3)

    data_preprocessing(train_data, "train", output_file)
    data_preprocessing(test_data, "test", output_file)
    data_preprocessing(validation_data, "valid", output_file)