import math
import numpy as np
import torch
import json
from torch.utils.data import (DataLoader, WeightedRandomSampler, SequentialSampler, TensorDataset)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, ReLU
import torch.nn.functional as F

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import os
from transformers import (WEIGHTS_NAME, BertConfig, BertModel, BertTokenizer,
                                  XLMConfig, XLMModel, XLMTokenizer, 
                                  XLNetConfig, XLNetModel, XLNetTokenizer,
                                  RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup

from data_processor import *

import pdb


processors = {
    "both": BinaryClassificationProcessor
}

output_modes = {
    "both": "classification|regression"
}


def load_and_cache_examples(task, tokenizer, evaluate=False, final=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'unlabeled_cikm'
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_unlabeled_pickles("/dgxhome/cra5302/Disclosure/TopicModeling/")
        
        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode)
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


args = {
    'data_dir': 'processed_data/',
    'model_type':  'bert',
    'model_name': 'bert-base-cased',
    'task_name': 'both',
    'output_dir': 'outputs_information_bert/',
    'cache_dir': 'cache/',
    'do_train': False,
    'do_eval': False,
    'do_predict': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'both',
    'train_batch_size': 16,
    'eval_batch_size': 150,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 4,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 100,
    'evaluate_during_training': True,
    'save_steps': 200,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'notes': 'Using Twitter disclosure dataset'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('args.json', 'w') as f:
    json.dump(args, f)


if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))



MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
    'xlm': (XLMConfig, XLMModel, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]


config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])


base_model = model_class.from_pretrained(args['model_name'])


@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
    def forward(self, input):
        return mish(input)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        # self.dense = nn.Linear(hidden_size, 192)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class ClassificationModel(nn.Module):
    def __init__(self, base_model, base_model_output_size=768, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        
        self.dropout = nn.Dropout(dropout)
        self._lambda = 1.0
        self.classifier = ClassificationHead(base_model_output_size, dropout, 2)
        self.regressor = nn.Linear(base_model_output_size, 1)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, binary_labels=None, level_labels=None, do_binary=True):
        X, attention_mask = input_ids, attention_mask
        outputs = self.base_model(X, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        binary_logits = self.classifier(pooled_output)
        # level_logit = self.regressor(pooled_output)
        level_logit = None

        if binary_labels is not None:
            outputs = (binary_logits, level_logit,) + outputs[2:]  # add hidden states and attention if they are here
            
            binary_loss_fct = CrossEntropyLoss()
            binary_loss = binary_loss_fct(binary_logits.view(-1, 2), binary_labels.view(-1))
            
            regression_loss_fct = MSELoss()
#             regression_loss = regression_loss_fct(
#                 level_logit[binary_labels==1, :].view(-1), 
#                 level_labels[binary_labels==1].view(-1)
#             )
            regression_loss = 0
            # pdb.set_trace()
            print("\r{:.5f} {:.5f}".format(binary_loss, regression_loss), end='')
            loss = binary_loss + self._lambda * regression_loss
            
            outputs = (loss,) + outputs
        else:
            # level_logits = torch.clamp(level_logit, min=0, max=1)
            outputs = (binary_logits, level_logit,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), binary_logits, level_logits, (hidden_states), (attentions)



model = ClassificationModel(base_model=base_model)
model.to(device);


task = args['task_name']

if task in processors.keys() and task in output_modes.keys():
    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
else:
    raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')



from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix, f1_score
from scipy.stats import pearsonr


def predict(model, tokenizer, prefix="", final=False):
    model.eval()
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    nb_eval_steps = 0
    binary_preds = None
    level_preds = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                     }
            outputs = model(**inputs)
            binary_logits, level_logit = outputs[:3]
        nb_eval_steps += 1
        if binary_preds is None:
            binary_preds = binary_logits.detach().cpu().numpy()
            # level_preds = level_logit.detach().cpu().numpy()
        else:
            binary_preds = np.append(binary_preds, binary_logits.detach().cpu().numpy(), axis=0)
            # level_preds = np.append(level_preds, level_logit.detach().cpu().numpy(), axis=0)
    
    binary_preds = np.argmax(binary_preds, axis=1)
    # level_preds = np.squeeze(level_preds)
    
    pdb.set_trace()
    
    return binary_preds, level_preds


if args['do_predict']:
    print ('\n\n')
    global_step = 200
    checkpoint = args['output_dir'] + '/checkpoint-{}/'.format(str(global_step)) + WEIGHTS_NAME
    model = ClassificationModel(base_model=base_model)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    binary_preds, _ = predict(model, tokenizer, prefix=global_step)
    
    # pdb.set_trace()
    import pickle as pkl
    
    with open('/dgxhome/cra5302/Disclosure/TopicModeling/tweet_txt.pkl', 'rb') as f:
        tweet_dict = pkl.load(f)
    
    label_dict = {}
    
    for idx, (id, _) in enumerate(tweet_dict.items()):
        label_dict[id] = binary_preds[idx]
    
    with open("label_dict.pkl", 'wb') as f:
        pkl.dump(label_dict, f)

