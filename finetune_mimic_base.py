### Required Packages
from termcolor import colored
import math
import pandas as pd
import random
import numpy as np
import pickle as pkl
import os


data_path = "/sent_bert/"
data_path_2 = "/Med-BERT/data/"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

use_cuda = torch.cuda.is_available()
from transformers import BertForSequenceClassification
import sys


# #### Load Data from pickled list
#
# The pickled list is a list of lists where each sublist represent a patient record that looks like
# [pt_id,label, seq_list , segment_list ]
# where
#     Label: 1: pt developed HF (case) , 0 control
#     seq_list: list of all medical codes in all visits
#     segment list: the visit number mapping to each code in the sequence list
#

### Below are key functions for  Data prepartion ,formating input data into features, and model defintion

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_EHRexamples_to_features(examples, max_seq_length):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_singleEHR_example(ex_index, example, max_seq_length)
        features.append(feature)
    return features


### This is the EHR version

def convert_singleEHR_example(ex_index, example, max_seq_length):
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    input_ids = example[2]
    segment_ids = example[3]
    label_id = example[1]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # LR 5/13 Left Truncate longer sequence
    while len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        input_mask = input_mask[-max_seq_length:]
        segment_ids = segment_ids[-max_seq_length:]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = [input_ids, input_mask, segment_ids, label_id, True]
    return feature


# In[32]:


class BERTdataEHR(Dataset):
    def __init__(self, Features):
        self.data = Features

    def __getitem__(self, idx, seeDescription=False):
        sample = self.data[idx]

        return sample

    def __len__(self):
        return len(self.data)

    # customized parts for EHRdataloader


def my_collate(batch):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in batch:
        all_input_ids.append(feature[0])
        all_input_mask.append(feature[1])
        all_segment_ids.append(feature[2])
        all_label_ids.append(feature[3])
    return [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]


class BERTdataEHRloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=4, collate_fn=my_collate, pin_memory=False, drop_last=False,
                            timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn


# ##### Model Definition

# In[41]:


class EHR_BERT_LR(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, n_layers=1, dropout_r=0.3, cell_type='LSTM', bi=False,
                 time=False, preTrainEmb=''):
        super(EHR_BERT_LR, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.preTrainEmb = preTrainEmb
        self.time = time

        if bi:
            self.bi = 2
        else:
            self.bi = 1

        self.PreBERTmodel = BertForSequenceClassification.from_pretrained(os.path.join(data_path))
        if use_cuda:
            self.PreBERTmodel.cuda()
        input_size = self.PreBERTmodel.bert.config.vocab_size
        self.in_size = self.PreBERTmodel.bert.config.hidden_size

        self.dropout = nn.Dropout(p=self.dropout_r)
        self.out = nn.Linear(self.in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if use_cuda:
            self.flt_typ = torch.cuda.FloatTensor
            self.lnt_typ = torch.cuda.LongTensor
        else:
            self.lnt_typ = torch.LongTensor
            self.flt_typ = torch.FloatTensor

    def forward(self, sequence):
        token_t = torch.from_numpy(np.asarray(sequence[0], dtype=int)).type(self.lnt_typ)
        seg_t = torch.from_numpy(np.asarray(sequence[2], dtype=int)).type(self.lnt_typ)
        Label_t = torch.from_numpy(np.asarray(sequence[3], dtype=int)).type(self.lnt_typ)
        Bert_out = self.PreBERTmodel.bert(input_ids=token_t,
                                          attention_mask=torch.from_numpy(np.asarray(sequence[1], dtype=int)).type(
                                              self.lnt_typ),
                                          token_type_ids=seg_t)
        output = self.sigmoid(self.out(Bert_out[1]))
        return output.squeeze(), Label_t.type(self.flt_typ)


# In[48]:


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainsample(sample, model, optimizer, criterion=nn.BCELoss()):
    model.train()
    model.zero_grad()
    output, label_tensor = model(sample)
    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


# train with loaders

def trainbatches(mbs_list, model, optimizer, shuffle=True):
    current_loss = 0
    all_losses = []
    plot_every = 5
    n_iter = 0
    if shuffle:
        random.shuffle(mbs_list)
    for i, batch in enumerate(mbs_list):
        output, loss = trainsample(batch, model, optimizer, criterion=nn.BCELoss())
        current_loss += loss
        n_iter += 1

        if n_iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return current_loss, all_losses


def calculate_auc(model, mbs_list, shuffle=True):
    model.eval()
    y_real = []
    y_hat = []
    if shuffle:
        random.shuffle(mbs_list)
    for i, batch in enumerate(mbs_list):
        output, label_tensor = model(batch)
        y_hat.extend(output.cpu().data.view(-1).numpy())
        y_real.extend(label_tensor.cpu().data.view(-1).numpy())
    auc = roc_auc_score(y_real, y_hat)
    avg_precision = average_precision_score(list(y_real), list(y_hat))
    pred = np.array(y_hat).copy()

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    f1 = f1_score(list(y_real), list(pred))
    acc = balanced_accuracy_score(list(y_real), list(pred))
    return auc, avg_precision, f1, acc, y_real, y_hat


# define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test1, model, optimizer, shuffle=True, patience=20, output_dir=data_path,
               model_prefix='dhf.train', model_customed=''):
    bestTrainAuc = 0.0

    bestValidAuc = 0.0
    bestTestAuc1 = 0.0
    bestValidEpoch = 0

    bestTestF1 = 0
    bestTestprecision = 0
    bestTestacc = 0

    # header = 'BestValidAUC|TestAUC|atEpoch'
    # logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    # print2file(header, logFile)
    # writer = SummaryWriter(output_dir+'/tsb_runs/') ## LR added 9/27 for tensorboard integration
    for ep in range(epochs):
        print(ep, flush=True)
        start = time.time()
        current_loss, train_loss = trainbatches(mbs_list=train, model=model, optimizer=optimizer)
        train_time = timeSince(start)
        # epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        # writer.add_scalar('Loss/train', avg_loss, ep) ## LR added 9/27
        valid_start = time.time()
        train_auc, train_avg_precision, train_f1, train_acc, _, _ = calculate_auc(model=model, mbs_list=train,
                                                                                  shuffle=shuffle)
        valid_auc, valid_avg_precision, valid_f1, valid_acc, _, _ = calculate_auc(model=model, mbs_list=valid,
                                                                                  shuffle=shuffle)
        valid_time = timeSince(valid_start)
        # writer.add_scalar('train_auc', train_auc, ep) ## LR added 9/27
        # writer.add_scalar('valid_auc', valid_auc, ep) ## LR added 9/27
        print(colored(
            '\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s)' % (
                ep, train_auc, valid_auc, avg_loss, train_time, valid_time), 'green'), flush=True)
        buf = '\n valid_auc:%f, valid_avg_precision:%f, valid_f1:%f, valid_acc:%f' % (
            valid_auc, valid_avg_precision, valid_f1, valid_acc)
        print(buf, flush=True)

        if valid_auc > bestValidAuc:
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            best_model = model
            bestTrainAuc = train_auc
            if test:
                testeval_start = time.time()
                bestTestAuc1, test_avg_precision, test_f1, test_acc, _, _ = calculate_auc(model=best_model,
                                                                                          mbs_list=test1,
                                                                                          shuffle=shuffle)
                bestTestF1 = test_f1
                bestTestprecision = test_avg_precision
                bestTestacc = test_acc
                # writer.add_scalar('test_auc', valid_auc, ep) ## LR added 9/27
                print(colored('\n Test_AUC1 (%s) , Test_eval_time (%s) ' % (bestTestAuc1, timeSince(testeval_start)),
                              'yellow'), flush=True)
                buf = 'Currently the parameters with best vald_auc: test_auc:%f, test_avg_precision:%f, test_f1:%f, test_acc%f' % (
                    bestTestAuc1, test_avg_precision, test_f1, test_acc)
                print(buf, flush=True)
                # print(best_model,model) ## to verify that the hyperparameters already impacting the model definition
                # print(optimizer)
        if ep - bestValidEpoch > patience:
            break

    # writer.close()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ## save model & parameters
    torch.save(best_model, output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    torch.save(best_model.state_dict(), output_dir + model_prefix + model_customed + 'EHRmodel.st')

    if test:
        print(colored(
            'BestValidAuc %f has parameters test_auc:%f, test_avg_precision:%f, test_f1:%f, test_acc%f at epoch %d ' % (
                bestValidAuc, bestTestAuc1, bestTestprecision, bestTestF1, bestTestacc, bestValidEpoch),
            'green'), flush=True)
        return bestTrainAuc, bestValidAuc, bestTestAuc1, bestValidEpoch, bestTestprecision, bestTestF1, bestTestacc
    else:
        print(colored('BestValidAuc %f at epoch %d ' % (bestValidAuc, bestValidEpoch), 'green'), flush=True)
        print('No Test Accuracy', flush=True)

    print(colored('Details see ../models/%sEHRmodel.log' % (model_prefix + model_customed), 'green'), flush=True)


MAX_SEQ_LENGTH = 128
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
bert_config_file = data_path + "config.json"

if __name__ == "__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    train_f = pkl.load(open(os.path.join(data_path, "MIMIC_new_task_finetune_" + inFile + ".bencs.train"), 'rb'),
                       encoding='bytes')
    valid_f = pkl.load(open(os.path.join(data_path, "MIMIC_new_task_finetune_" + inFile + ".bencs.valid"), 'rb'),
                       encoding='bytes')
    test_f = pkl.load(open(os.path.join(data_path, "MIMIC_new_task_finetune_" + inFile + ".bencs.test"), 'rb'), encoding='bytes')

    results = []

    #### Data Preparation
    train_features = convert_EHRexamples_to_features(train_f, MAX_SEQ_LENGTH)
    test_features = convert_EHRexamples_to_features(test_f, MAX_SEQ_LENGTH)
    valid_features = convert_EHRexamples_to_features(valid_f, MAX_SEQ_LENGTH)

    train = BERTdataEHR(train_features)
    test = BERTdataEHR(test_features)
    valid = BERTdataEHR(valid_features)

    print(' creating the list of training minibatches', flush=True)
    train_mbs = list(BERTdataEHRloader(train, batch_size=BATCH_SIZE))
    print(' creating the list of test minibatches', flush=True)
    test_mbs = list(BERTdataEHRloader(test, batch_size=BATCH_SIZE))
    print(' creating the list of valid minibatches', flush=True)
    valid_mbs = list(BERTdataEHRloader(valid, batch_size=BATCH_SIZE))

    for run in range(5):  ### to average the results on 10 runs
        for model_type in ['Bert only']:
            ehr_model = EHR_BERT_LR(input_size=28624, embed_dim=384, hidden_size=384)
            # print(ehr_model, flush=True)
            # for name, p in ehr_model.named_parameters():
            #     print(name, p, flush=True)
            ehr_model.PreBERTmodel.bert.embeddings.word_embeddings.weight.requires_grad = False

            # for name, p in ehr_model.named_parameters():
            #     if p.requires_grad:
            #         print(name, p)
            # print(ehr_model.PreBERTmodel.bert.embeddings.word_embeddings.weight)
            # print(ehr_model.PreBERTmodel.bert.embeddings.word_embeddings.weight.shape)

            if use_cuda:
                ehr_model.cuda()
            optimizer = optim.Adam(ehr_model.parameters(), lr=LEARNING_RATE)
            out_dir_name = outFile + "_MIMIC"
            trauc, vauc, testauc1, bep, bestTestprecision, bestTestF1, bestTestacc = epochs_run(100, train=train_mbs,
                                                                                                valid=valid_mbs,
                                                                                                test1=test_mbs,
                                                                                                model=ehr_model,
                                                                                                optimizer=optimizer,
                                                                                                shuffle=True,
                                                                                                patience=10,
                                                                                                output_dir=out_dir_name,
                                                                                                model_prefix=outFile + '_first_run_')
            results.append(
                [model_type, run, len(train_features), len(test_features), len(valid_features), trauc, vauc, testauc1,
                 bep, bestTestprecision, bestTestF1, bestTestacc])

    df = pd.DataFrame(results)
    df.columns = ['Model', 'Run', 'Train_size', 'Test_size', 'Valid_size', 'Train_AUC', 'Valid_AUC', 'Test_AUC1',
                  'Best_Epoch', "BestTestPrecision", "BestTestF1", "BestTestAcc"]
    print(df, flush=True)
    df.to_csv("Final_results_MIMIC_finetuning_" + outFile + ".csv", index=False)

    desc2 = df[['Model', 'Train_size', 'Test_AUC1']].groupby(['Model', 'Train_size']).describe()
    print(desc2, flush=True)
