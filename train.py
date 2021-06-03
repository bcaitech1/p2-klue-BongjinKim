import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import FunnelForSequenceClassification, FunnelConfig, AdamW, FunnelTokenizerFast, ElectraTokenizer, ElectraModel, BertTokenizer, BertModel, AlbertModel, FunnelTokenizer, FunnelModel, AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizerFast, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig, BertConfig
from load_data import *
import argparse
from importlib import import_module
import wandb
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
import time
import datetime
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import numpy as np
from loss import create_criterion
from adamp import AdamP, SGDP
import copy
from sklearn.model_selection import KFold

hyperparameter_defaults = dict(
    dropout = 0.1,
    batch_size = 32,
    learning_rate = 1e-5,
    epochs = 8,
    model_name = 'XLMRobertaForSequenceClassification',
    tokenizer_name = 'XLMRobertaTokenizer',
    config_name = 'XLMRobertaConfig',
    seed = 42,
    )

wandb.init(config=hyperparameter_defaults, project="bert-test")
config = wandb.config
weights = None
def compute_weight(labels):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    df = pd.read_csv('/opt/ml/input/data/train/train.tsv', sep='\t', names=[1,2,3,4,5,6,7,8,'label'])
    
    label = copy.deepcopy(label_type)
    for i in label:
        label[i]=0
    for ele in df.label:
        label[ele] += 1

    total=len(df.values)
    arr = [total/label[key] for i, key in enumerate(label)]
    return arr
    

class MultilabelTrainer(Trainer):
    #loss 재정의 가능
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        criterion = create_criterion('label_smoothing')
        loss = criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

def find_path(name):
    if name == 'BertForSequenceClassification' or name == 'BertTokenizerFast' or name == 'BertConfig':
        path = 'kykim/bert-kor-base'
    elif name == 'AlbertForSequenceClassification' or name == 'AlbertConfig':
        path = 'kykim/albert-kor-base'
    elif name == 'FunnelForSequenceClassification' or name == 'FunnelTokenizer' or name == 'FunnelConfig':
        path = 'kykim/funnel-kor-base'
    else:
        path = 'xlm-roberta-large'
    return path

def train():
#     with wandb.init(config=sweep_config):
#         # --> sweep controller
#         config = wandb.config
        
    # --> load model and tokenizer
    load_tokenizer = getattr(import_module("transformers"), config.tokenizer_name).from_pretrained
    tokenizer = load_tokenizer(find_path(config.tokenizer_name))
    tokenizer.add_special_tokens({"additional_special_tokens": [" @ ", " α ", ' # ', ' β ']})
    # load dataset
    dataset = load_data("/opt/ml/input/data/train/ner_train_ver2.tsv")
    #dataset2 = load_data("/opt/ml/input/data/train/train_weighted_data.tsv")
    
#     #weights = compute_weight(labels)
    #dataset = pd.concat([dataset1, dataset2])
    
    # prepare cross validation
    n=5
    kf = KFold(n_splits=n, random_state=config.seed, shuffle=True)
    results = []
    # random_split으로 train, val data 나눔
#     train_size = int(0.8 * len(RE_dataset))
#     val_size = len(RE_dataset) - train_size
#     train_dataset, val_dataset = random_split(RE_dataset, [train_size, val_size])
    
    for idx, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_df = dataset.iloc[train_index]
        val_df = dataset.iloc[val_index]

        train_labels = train_df['label'].values
        val_labels = val_df['label'].values
                       
        # tokenizing dataset
        train_tokenized_datas = tokenized_dataset(train_df, tokenizer)
        val_tokenized_datas = tokenized_dataset(val_df, tokenizer)

        # make dataset for pytorch.
        #RE_dataset = RE_Dataset(tokenized_datas, np.concatenate((labels, val_labels), axis=0))
        train_dataset = RE_Dataset(train_tokenized_datas, train_labels)
        val_dataset = RE_Dataset(val_tokenized_datas, val_labels)                              
                              
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
        load_config = getattr(import_module("transformers"), config.config_name).from_pretrained
        bert_config = load_config(find_path(config.model_name))
        bert_config.num_labels = 42
        #bert_config.hidden_size = 1024
        #bert_config.embedding_size = 1024
        #print(bert_config)
        
        load_model = getattr(import_module("transformers"), config.model_name).from_pretrained
        model = load_model(find_path(config.model_name), config=bert_config) 

        model.to(device)
        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            seed = config.seed,
            fp16=True,
            dataloader_num_workers=4,
            output_dir= f'./results/k-fold-{idx}',          # output directory
            save_total_limit=1,              # number of total save model.
            save_steps=100,                 # model saving step.
            num_train_epochs=config.epochs,              # total number of training epochs
            learning_rate=config.learning_rate,               # learning_rate
            per_device_train_batch_size=config.batch_size,  # batch size per device during training
            per_device_eval_batch_size=config.batch_size,   # batch size for evaluation
            warmup_steps=300,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_steps=100,              # log saving step.
            evaluation_strategy='steps',
            eval_steps = 100,            
            report_to = 'wandb',
            load_best_model_at_end = True,
            metric_for_best_model = 'accuracy',
    #         run_name = 'custom_training'            # name of the W&B run
        )
        #os.environ['WANDB_WATCH'] = 'all'
        trainer = MultilabelTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,         # define metrics function
            tokenizer=tokenizer,
        )

        # train model
        trainer.train()
        #trainer.evaluate()
    
def main():
    train()

if __name__ == '__main__' :
    main()
