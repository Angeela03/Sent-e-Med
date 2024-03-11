# Sent-e-Med

This is the implementation of the Sent-e-Med model from the paper "Clinical Risk Prediction Using Language Models: Benefits and Considerations" https://arxiv.org/abs/2312.03742. The code has been adapted from https://github.com/ZhiGroup/Med-BERT

finetune_mimic_base.py- Finetune the Sent-e-Med model
Preprocess_for_finetune_mimic.py - Preprocess data for finetuning
run_EHRpretraining_QA2Seq.py - Code for pretraining the Sent-e-Med model
tensorflow_2_pytorch.py - Convert tensorflow code to pytorch 
process_embeddings_sent_bert.py - Process SBERT embeddings to use in the model
