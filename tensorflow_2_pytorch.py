# Convert the tensorflow model to pytorch for finetuning

import torch
import os
from transformers import BertConfig, BertModel, BertTokenizer, BertForPreTraining
import torch
import torch.nn as nn
import logging
import numpy as np
from transformers import BertForSequenceClassification
import tensorflow as tf
use_cuda = torch.cuda.is_available()
# data_path = "/Med-BERT/medbert_new_task/sent_bert"

embedding_path="token_embeddings.npy"


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        # logger.error(
        #     "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        #     "https://www.tensorflow.org/install/ for installation instructions."
        # )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    # logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # logger.info(f"Loading TF weight {name} with shape {shape}")
        print(name, shape)
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    # logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)
    # print(model)

    model.cls.seq_relationship = nn.Linear(config.hidden_size, config.vocab_label_size)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    embedding_table = np.load(embedding_path, allow_pickle=True)
    embedding_table = torch.from_numpy(embedding_table)

    # print(type(model.bert.embeddings.word_embeddings))
    model.bert.embeddings.word_embeddings = nn.Embedding.from_pretrained(embedding_table)

    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


convert_tf_checkpoint_to_pytorch("model.ckpt-1000000", "config.json", "pytorch_model.bin")