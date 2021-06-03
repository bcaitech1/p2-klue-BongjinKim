from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from transformers import FunnelForSequenceClassification, FunnelConfig, AdamW, ElectraTokenizerFast, ElectraModel, BertTokenizerFast, BertModel, AlbertModel, FunnelTokenizer, FunnelModel, AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from load_data import *
from importlib import import_module
from tqdm import tqdm
def class2label(preds):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    trans = {label_type[k]:k for k in label_type}
    return [trans[pred] for pred in preds]

def find_path(name):
    if name == 'BertForSequenceClassification' or name == 'BertTokenizerFast':
        path = 'kykim/bert-kor-base'
    elif name == 'AlbertForSequenceClassification':
        path = 'kykim/albert-kor-base'
    elif name == 'FunnelForSequenceClassification' or name == 'FunnelTokenizer':
        path = 'kykim/funnel-kor-base'
    else:
        path = 'xlm-roberta-large'
    return path

def inference(args, tokenized_sent, device):
    models=[]
    for i in range(args.k_num):
        load_model = getattr(import_module("transformers"), args.model_name).from_pretrained
        models.append(load_model(
            f"{args.model_path}-{i}/checkpoint", # 한국어로 학습된 Funnel model
            #args.model_path,
            num_labels = 42, # label의 수 42
            output_attentions = False, # attention 반환 여부
            output_hidden_states = False, # hiddenstate 반환 여부
        ))
    #     modelparameers
    dataloader = DataLoader(tokenized_sent, batch_size=20, shuffle=False, num_workers=4)
    output_pred = []
    logit_result = []
    for i, data in tqdm(enumerate(dataloader)):
        k_preds = []
        for model in models:
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                if 'token_type_ids' in data.keys():
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device)
                    )
                else:
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device)
                    )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            logits = np.expand_dims(logits, axis=0)
            #result = class--2label(result)
            k_preds.append(logits)
        #print(k_preds)
        result = np.argmax(np.mean(k_preds, axis=0), axis=-1)
        output_pred.append(result)
        logit_result.append(np.mean(k_preds, axis=0))
    return np.array(output_pred).flatten(), np.array(logit_result)

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def make_dataset(dataset_dir, preds):
    test_dataset = load_data(dataset_dir)
    test_dataset['label'] = preds
    # tokenizing dataset
    return test_dataset

def main(args):
    """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    load_tokenizer = getattr(import_module("transformers"), args.tokenizer_name).from_pretrained
    tokenizer = load_tokenizer(find_path(args.tokenizer_name))
    tokenizer.add_special_tokens({"additional_special_tokens": [" @ ", " α ", ' # ', ' β ']})

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    result, logit_result = inference(args, test_dataset, device)
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(result, columns=['pred'])
    
    np.save(os.path.join(args.logit_path, r'logits.npy'), logit_result)
    output.to_csv('/opt/ml/code/prediction/submission.csv', index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_name', type=str, default='FunnelForSequenceClassification', help='model name')
    parser.add_argument('--tokenizer_name', type=str, default='FunnelTokenizer', help='tokenier name')
    parser.add_argument('--model_path', type=str, default="", help='model path')
    parser.add_argument('--logit_path', type=str, default="/opt/ml/code/logits", help='logit path')
    parser.add_argument('--k_num', type=int, default=5, help='k number')
    args = parser.parse_args()   
    main(args)
  
