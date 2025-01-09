import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt 
from tensorboardX import SummaryWriter
import os

import json
import pandas as pd
from tqdm import tqdm



def usefulness_metrics(y_true, y_pred):
    all_metrics = {}

    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        all_metrics['auc'] = -1
    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['spauc'] = -1
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    try:
        all_metrics['f1_unuseful'], all_metrics['f1_useful'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['f1_unuseful'], all_metrics['f1_useful'] = -1, -1
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    try:
        all_metrics['recall_unuseful'], all_metrics['recall_useful'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['recall_unuseful'], all_metrics['recall_useful'] = -1, -1
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    try:
        all_metrics['precision_unuseful'], all_metrics['precision_useful'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['precision_unuseful'], all_metrics['precision_useful'] = -1, -1
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    return all_metrics



def metrics(y_true, y_pred):
    all_metrics = {}

    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        all_metrics['auc'] = -1
    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['spauc'] = -1
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    try:
        all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['f1_real'], all_metrics['f1_fake'] = -1, -1
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    try:
        all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['recall_real'], all_metrics['recall_fake'] = -1, -1
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    try:
        all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['precision_real'], all_metrics['precision_fake'] = -1, -1
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    return all_metrics


def llm_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    recall_real, recall_fake, _ = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    precision = precision_score(y_true, y_pred, average='macro')
    precision_real, precision_fake, _ = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {'acc': acc, 'f1': f1, 'recall': recall, 'precision': precision, 'precision_real': precision_real,
            'precision_fake': precision_fake, 'recall_real': recall_real, 'recall_fake': recall_fake}


class ARGMetricsRecorder:


    def __init__(self):
        self.classifier_labels = []
        self.classifier_predictions = []

        self.llm_judgment_labels = []
        self.llm_judgment_predictions = []

        self.rationale_usefulness_labels = []
        self.rationale_usefulness_predictions = []

    def record(self,batch_data,res):
        batch_label = batch_data['label']
        self.classifier_labels.append(batch_label)
        self.classifier_predictions.append(res['classify_pred'])
        self.llm_judgment_labels.append(torch.cat((batch_data['FTR_2_pred'], batch_data['FTR_3_pred']), dim=0))
        llm_judgment_prediction = torch.cat(
            (res['simple_ftr_2_pred'].argmax(dim=-1), res['simple_ftr_3_pred'].argmax(dim=-1))
            ,dim=0)
        self.llm_judgment_predictions.append(llm_judgment_prediction)
        self.rationale_usefulness_labels.append(torch.cat((batch_data['FTR_2_acc'], batch_data['FTR_3_acc']), dim=0))
        self.rationale_usefulness_predictions.append(torch.cat((res['hard_ftr_2_pred'], res['hard_ftr_3_pred']), dim=0))


    def get_metrics(self):
        self.classifier_labels = torch.cat(self.classifier_labels, dim=0).cpu().numpy()
        self.classifier_predictions = torch.cat(self.classifier_predictions, dim=0).cpu().numpy()
        self.llm_judgment_labels = torch.cat(self.llm_judgment_labels, dim=0).cpu().numpy()
        self.llm_judgment_predictions = torch.cat(self.llm_judgment_predictions, dim=0).cpu().numpy()
        self.rationale_usefulness_labels = torch.cat(self.rationale_usefulness_labels, dim=0).cpu().numpy()
        self.rationale_usefulness_predictions = torch.cat(self.rationale_usefulness_predictions, dim=0).cpu().numpy()

        classifier_metrics = metrics(self.classifier_labels, self.classifier_predictions)
        llm_judgment_metrics = llm_metrics(self.llm_judgment_labels, self.llm_judgment_predictions)
        rationale_usefulness_metrics = usefulness_metrics(self.rationale_usefulness_labels, self.rationale_usefulness_predictions)

        return {
            'classifier': classifier_metrics,
            'llm_judgment': llm_judgment_metrics,
            'rationale_usefulness': rationale_usefulness_metrics
        }

def try_all_gpus():
    return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


def data2gpu(batch, use_cuda, data_type):
    if use_cuda:
        return {
            k:batch[k].cuda(0)
            for k in batch.keys() if batch[k] is not None and isinstance(batch[k], torch.Tensor)
        }
    return batch

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)




class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_monthly_path(data_type, root_path, month, data_name):
    if data_type == 'rationale':
        file_path = os.path.join(root_path, data_name)
        return file_path
    else:
        print('No match data type!')
        exit()

def get_tensorboard_writer(config):
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    test_result = []
    test_df = pd.read_json(test_file_path)
    for index in range(len(label)):
        cur_res = {}
        cur_id = id[index]
        cur_data = test_df[test_df['id'] == int(cur_id)].iloc[0]
        for (key, val) in cur_data.iteritems(): 
            cur_res[key] = val
        cur_res['pred'] = pred[index]
        cur_res['ae'] = ae[index]
        cur_res['acc'] = acc[index]

        test_result.append(cur_res)

    json_str = json.dumps(test_result, indent=4, ensure_ascii=False, cls=NpEncoder)

    with open(test_res_path, 'w') as f:
        f.write(json_str)
    return
