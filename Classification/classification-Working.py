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
                                  RobertaConfig, RobertaModel, RobertaTokenizer,
                                  DistilBertConfig, DistilBertModel, DistilBertTokenizer
                        )
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
    
    mode = 'dev' if (evaluate and not final) else ('test' if (evaluate and final) else 'train')
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if (evaluate and not final) else\
                    (processor.get_test_examples(args['data_dir']) if (evaluate and final) else\
                     processor.get_train_examples(args['data_dir']))
        
        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode)
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "both":
        all_binary_labels = torch.tensor([f.binary_label for f in features], dtype=torch.long)
        all_level_labels = torch.tensor([f.level_label for f in features], dtype=torch.float)
    else:
        raise KeyError(output_mode)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_binary_labels, all_level_labels)
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
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'both',
    'train_batch_size': 16,
    'eval_batch_size': 128,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
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
    'reprocess_input_data': True,
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
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertModel, DistilBertTokenizer)
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
        outputs = self.base_model(X, attention_mask=attention_mask, 
                                  token_type_ids=token_type_ids
                                 )
        
        pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        binary_logits = self.classifier(pooled_output)
        level_logit = self.regressor(pooled_output)

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
            level_logits = torch.clamp(level_logit, min=0, max=1)
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

    
def get_weights_for_sampler(dataset):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=len(dataset))
    batch = None
    for _batch in dataloader:
        batch = _batch
    labels = batch[3].numpy()
    num_neg = len(labels[labels == 0])
    num_pos = len(labels[labels == 1])
    weights = [1./num_neg if l == 0 else 1./num_pos for l in labels]
    return weights
    

def train(train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    
    weights = get_weights_for_sampler(train_dataset)
    train_sampler = WeightedRandomSampler(num_samples=len(train_dataset), weights=weights)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total*2)
    
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for ee in train_iterator:
        logger.info('\n\n')
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'binary_labels':  batch[3],
                      'level_labels':   batch[4],
                      'do_binary':      ee == 0}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)
                    torch.save(model_to_save.state_dict(), output_dir + '/' + WEIGHTS_NAME)
                    logger.info("Saving model checkpoint to %s", output_dir)


    return global_step, tr_loss / global_step



from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix, f1_score
from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    f1 = f1_score(labels, preds)
    return {
        "mcc": mcc,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }, get_mismatched(labels, preds)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def get_eval_report_reg(labels, preds):
    rmse = mean_squared_error(labels, preds, squared=False)
    return {
        "rmse": rmse
    }

def compute_regression_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report_reg(labels, preds)

def evaluate(model, tokenizer, prefix="", final=False):
    model.eval()
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True, final=final)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    binary_preds = None
    level_preds = None
    out_binary_labels = None
    out_level_labels = None
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'binary_labels':  batch[3],
                      'level_labels':   batch[4]}
            outputs = model(**inputs)
            tmp_eval_loss, binary_logits, level_logit = outputs[:3]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if binary_preds is None:
            binary_preds = binary_logits.detach().cpu().numpy()
            out_binary_labels = inputs['binary_labels'].detach().cpu().numpy()
            
            level_preds = level_logit.detach().cpu().numpy()
            out_level_labels = inputs['level_labels'].detach().cpu().numpy()
        
        else:
            binary_preds = np.append(binary_preds, binary_logits.detach().cpu().numpy(), axis=0)
            out_binary_labels = np.append(out_binary_labels, inputs['binary_labels'].detach().cpu().numpy(), axis=0)
        
            level_preds = np.append(level_preds, level_logit.detach().cpu().numpy(), axis=0)
            out_level_labels = np.append(out_level_labels, inputs['level_labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    
    binary_preds = np.argmax(binary_preds, axis=1)
    level_preds = np.squeeze(level_preds)
    
    result, wrong = compute_metrics(EVAL_TASK, binary_preds, out_binary_labels)
    results.update(result)
    
    pdb.set_trace()
    
    result = compute_regression_metrics(EVAL_TASK, level_preds[out_binary_labels==1], out_level_labels[out_binary_labels==1])
    results = dict(list(results.items()) + list(result.items()))

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results, wrong



if args['do_train']:
    train_dataset = load_and_cache_examples(task, tokenizer)
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if args['do_train']:
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(args['output_dir'])
    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model_to_save.state_dict(), output_dir + '/' + WEIGHTS_NAME)
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))



import glob

results = {}
if args['do_eval']:
    print ('\n\n')
    checkpoints = [args['output_dir']]
    if args['eval_all_checkpoints']:
        # checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        checkpoints = list(c for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = ClassificationModel(base_model=base_model)
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
        result, wrong_preds = evaluate(model, tokenizer, prefix=global_step, final=True)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
        
print (results)
