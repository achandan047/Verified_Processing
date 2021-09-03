import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  


import json
import random
import re
import numpy as np
from os import path
import pdb
import pickle as pkl
import sys
from tqdm import tqdm
import numpy as np
import yaml
from datetime import datetime
import coloredlogs
import logging
from sklearn.metrics import f1_score

from sesame.evaluation import *
from sesame.dataio import *
from sesame.globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL, FRAME_DIR, EMPTY_FE
from sesame.conll09 import lock_dicts, post_train_lock_dicts
from sesame.housekeeping import filter_long_ex, merge_span_ex
from sesame.evaluation import calc_f, evaluate_example_argid, evaluate_corpus_argid

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, Adafactor
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


torch.manual_seed(42)
random.seed(42)


# from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT
# from .dataio import create_target_lu_map, get_wvec_map, read_conll
# from .evaluation import calc_f, evaluate_example_targetid
# from .frame_semantic_graph import LexicalUnit
# from .housekeeping import unk_replace_tokens
# from .raw_data import make_data_instance
# from .semafor_evaluation import convert_conll_to_frame_elements

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hier", type=bool, default=False, help="Use phrase structure features.")
args = parser.parse_args("")


with open('config_arg.yaml', 'r') as f:
    config = yaml.safe_load(f)
print ()
print (config)
print ()


train_conll = TRAIN_FTE

train_examples, _, _ = read_conll(train_conll)
post_train_lock_dicts()
frmfemap, corefrmfemap, _ = read_frame_maps()
if args.hier:
    frmrelmap, feparents = read_frame_relations()
lock_dicts()

NOTANFEID = FEDICT.getid(EMPTY_FE)  # O in CoNLL format.

print ("Total number of distinct frames", FRAMEDICT.size())
print ("Total number of dictinct frame elements", FEDICT.size())

with open(config['frame_predfile'], 'rb') as handle:
    infer_examples = np.load(handle)


MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


class DataInstance:
    def __init__(self, guid, task, ex, frame_label,
                 input_ids, input_mask,
                 target_ids, target_mask):
        self.guid = guid
        self.task = task
        self.ex = ex
        self.frame_label = frame_label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.target_ids = target_ids
        self.target_mask = target_mask


class CustomDataset(Dataset):
    def __init__(self, examples):
        self.instances = self.process_instances(examples)
    
    def sent_format(self, words, targetpos):
        sent = ""
        for index, word in enumerate(words):
            sent += " " + str(index)
            if index != targetpos: sent += " " + word
            else: sent += " * " + word + " *"
        return sent
    
    def frame_format(self, frame, words, targetpos):
        frame_str = " ".join(frame.get_str(FRAMEDICT).split('_'))
        sent = "FRAME:" + self.sent_format(words, targetpos)
        return sent, frame_str
    
    def arg_format(self, frame, framedict, words, targetpos):
        frame_str = " ".join(frame.get_str(FRAMEDICT).split('_'))
        sent = "ARGS for " + frame_str + ":" + self.sent_format(words, targetpos)
        arg_str = ""
        for fe, spans in framedict.items():
            if fe == NOTANFEID: continue
            fe_str = " ".join(FEDICT.getstr(fe).split('_'))
            span = spans[0]
            arg_str += " " + fe_str + " = " + str(span[0]) + "-" + str(span[1]) + " |"
        return sent, arg_str
    
    def process_instances(self, examples):
        instances = []
        guid = 0
        max_seq_length = 128
        
        # pdb.set_trace()
        
        for index, ex in enumerate(tqdm(examples)):
            words = [VOCDICT.getstr(tok) for tok in ex.tokens]
            frame = ex.frame
            tfdict = ex.targetframedict
            
            targetpos = [i for i in range(len(words)) if ex._elements[i].is_pred][0]
            
            arg_inputstr, _ = self.arg_format(ex.frame, ex._get_inverted_femap(), words, targetpos)
            arg_inputenc = tokenizer.encode_plus(arg_inputstr, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='np')
            instances.append(
                DataInstance(index, task, ex, None,
                            arg_inputenc['input_ids'][0], arg_inputenc['attention_mask'][0],
                            None, None
            )
        
        return instances

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        return (self.instances[index].input_ids, self.instances[index].input_mask,
                self.instances[index].target_ids, self.instances[index].target_mask)


infer_arg_dataset = CustomDataset(infer_examples)
infer_arg_dataloader = torch.utils.data.DataLoader(infer_arg_dataset, batch_size=config['batch_size'], shuffle=True)

# # Model

import collections

def make_invertedfedict(arg_txt):
    fedict = {}
    
    args = arg_txt.split("|")
    for arg in args:
        if arg == '': continue
        fename, span = arg.split("=")
        fename, span = fename.strip(), span.strip()
        fename = "_".join(fename.split(" "))
        feid = FEDICT.getid(fename)
        span = span.split("-")
        spanlist = [(int(span[0]), int(span[1]))]
        fedict[feid] = spanlist
        
    return fedict



logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        
class Argloader:
    def __init__(self, dl1, shuffle=True):
        self.len = len(dl1)
        self.arg_data = [t for t in dl1]
        if shuffle:
            random.shuffle(self.arg_data)
        self.index = -1
        
    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        self.index += 1
        if self.index >= self.len:
            raise StopIteration
        return self.arg_data[self.index]


class SeqSupervisedNetwork(nn.Module):
    def __init__(self, config):
        super(SeqSupervisedNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        self.learner = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        # self.learner = torch.nn.DataParallel(self.learner)
        
        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))
        
        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        logger.info('Model loaded to {}'.format(self.device))
        
        self.initialize_optimizer_scheduler()
        
        
    def initialize_optimizer_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.learner.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in self.learner.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)
        # self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)
        
        self.optimizer = Adafactor(optimizer_grouped_parameters, relative_step=True, warmup_init=True, lr=None)

        
    def vectorize(self, batch):
        with torch.no_grad():
            if len(batch) == 3:
                return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2])
            else:
                return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3])
        
        
    def arg_forward(self, input_ids, input_mask, target_ids=None, target_mask=None):
        if target_ids is None:
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "max_length": 128, "num_beams": 2}
            outputs = self.learner.generate(**inputs)
            return outputs
        else:
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "labels": target_ids, "decoder_attention_mask": target_mask}
            outputs = self.learner(**inputs)
            loss = outputs[0]
            return loss
    
    def forward_eval(self, arg_dataloader, mode):
        if mode == 1:
            examples = dev_examples
        elif mode == 2:
            examples = test_examples
        elif mode == 3:
            examples = train_examples
        
        self.eval()
        
        lmetrics = umetrics = tokmetrics = [0., 0., 0.]
        labels, preds = [], []
        with torch.no_grad():
            for batch in tqdm(arg_dataloader, desc="Arg evaluation"):
                batch = self.vectorize(batch)
                batch = tuple(t.to(config['device']) for t in batch)
                output = self.arg_forward(batch[0], batch[1])
                pred_txt = tokenizer.batch_decode(output)
                label_txt = tokenizer.batch_decode(batch[2])
                preds.extend(pred_txt)
                labels.extend(label_txt)
            
            for index in range(len(preds)):
                try:
                    prediction = make_invertedfedict(preds[index])
                    u, l, w = evaluate_example_argid(examples[index]._get_inverted_femap(), prediction,
                                                 corefrmfemap[examples[index].frame.id],
                                                 len(examples[index].tokens), NOTANFEID)
                except:
                    # pdb.set_trace()
                    # logger.error(preds[index])
                    continue

                umetrics = np.add(umetrics, u)
                lmetrics = np.add(lmetrics, l)
                tokmetrics = np.add(tokmetrics, w)

        _, _, uf1_score = calc_f(umetrics)
        lprec, lrec, lf1_score = calc_f(lmetrics)
        _, _, tokf1 = calc_f(tokmetrics)
        
        return lprec, lrec, lf1_score
                
    
    def forward_infer(self, arg_dataloader):
        self.eval()
        
        preds_txt = []
        predictions = []
        with torch.no_grad():
            for batch in tqdm(arg_dataloader, desc="Arg evaluation"):
                batch = self.vectorize(batch)
                batch = tuple(t.to(config['device']) for t in batch)
                output = self.arg_forward(batch[0], batch[1])
                pred_txt = tokenizer.batch_decode(output)
                label_txt = tokenizer.batch_decode(batch[2])
                preds_txt.extend(pred_txt)
            
            for index in range(len(preds)):
                try:
                    prediction = make_invertedfedict(preds[index])
                    predictions.append(prediction)
                except:
                    # pdb.set_trace()
                    # logger.error(preds[index])
                    continue

        return predictions
        
    
    def forward(self, arg_dataloader, mode):
        if mode != 0:
            if mode < 4:
                return self.forward_eval(arg_dataloader, mode)
            else:
                return self.forward_infer(arg_dataloader)
        
        self.train()
        
        dataloader = Argloader(arg_dataloader)
        
        log_every = 100
        avg_arg_loss = 0
        loss_log = 0
        
        loader_t = tqdm(iter(dataloader))
        for batch_id, batch in enumerate(loader_t):
            batch = self.vectorize(batch)
            batch = tuple(t.to(config['device']) for t in batch)
            
            loss = self.arg_forward(batch[0], batch[1], batch[2])
            loss = loss.mean()
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            
            avg_arg_loss += loss.item() * self.gradient_accumulation_steps

            if (batch_id + 1) % self.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(self.learner.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.lr_scheduler.step()
            
            if (batch_id + 1) % log_every == 0:
                avg_arg_loss /= log_every
                loss_log = avg_arg_loss
                loader_t.set_description('Batch {}/{}, avg_argloss = {:.5f}'.format(batch_id + 1, len(dataloader), avg_arg_loss))
                loader_t.refresh()
                avg_arg_loss = 0

        return loss_log

# In[33]:


class SupervisedNetwork:
    def __init__(self, config):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.tensorboard_writer = SummaryWriter(log_dir='runs/{}-'.format(MODEL_NAME) + date_time)
        
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.meta_epochs = config['num_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)

        self.model = SeqSupervisedNetwork(config)

        logger.info('Supervised network instantiated')

    def training(self, train_arg_dataloader, val_arg_dataloader):
        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', '{}-t5model.h5'.format(self.stamp))
        logger.info('Model name: {}-t5model.h5'.format(self.stamp))
        for epoch in range(self.meta_epochs):
            logger.info('Starting epoch {}/{}'.format(epoch + 1, self.meta_epochs))
            
            avg_loss = self.model(train_arg_dataloader, mode=0)

            logger.info('Train epoch {}: Avg loss = {:.5f}'.format(epoch + 1, avg_loss))
            self.tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch + 1)
            
            avg_precision, avg_recall, avg_f1 = self.model(val_arg_dataloader, mode=1)

            logger.info('Val epoch {}: avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_precision, avg_recall, avg_f1))
            
            if avg_f1 > best_f1 + self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                logger.info('Saving the model since the F1 improved')
                torch.save(self.model.learner.state_dict(), model_path)
                logger.info('')
            else:
                patience += 1
                logger.info('F1 did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break

            # Log params and grads into tensorboard
#             for name, param in self.model.named_parameters():
#                 if param.requires_grad and param.grad is not None:
#                     self.tensorboard_writer.add_histogram('Params/' + name, param.data.view(-1),
#                                                      global_step=epoch + 1)
#                     self.tensorboard_writer.add_histogram('Grads/' + name, param.grad.data.view(-1),
#                                                      global_step=epoch + 1)

        self.model.learner.load_state_dict(torch.load(model_path))
        return best_f1

    def testing(self, test_arg_dataloader):
        logger.info('---------- Supervised testing starts here ----------')
        precision, recall, f1_score = self.model(test_arg_dataloader, mode=2)

        logger.info('Avg meta-testing metrics: precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(precision,
                                               recall,
                                               f1_score))
        return f1_score
    
    def testing_train(self, arg_dataloader):
        logger.info('---------- Evaluating training data starts here ----------')
        precision, recall, f1_score = self.model(arg_dataloader, mode=3)

        logger.info('Avg meta-testing metrics: precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(precision,
                                               recall,
                                               f1_score))
        return f1_score
    
    def inference(self, arg_dataloader):
        logger.info('---------- Evaluating training data starts here ----------')
        predictions = self.model(arg_dataloader, mode=4)

        with open('arg_predictions.npy', 'wb') as handle:
            np.save(handle, predictions)


# In[34]:


learner = SupervisedNetwork(config)


# In[35]:


if config['train']:
    learner.training(train_arg_dataloader, dev_arg_dataloader)
    logger.info('Supervised learning completed')


# In[ ]:


learner.testing(test_arg_dataloader)
logger.info('Supervised testing completed')


train_arg_dataloader = torch.utils.data.DataLoader(train_arg_dataset, batch_size=config['batch_size'], shuffle=False)
learner.testing_train(train_arg_dataloader)
logger.info('Is there overfit?')