import json
import os

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel

from models.arg import DualCrossAttentionFeatureAggregator
from utils.dataloader import load_data
from utils.utils import data2gpu, try_all_gpus

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import os



import logging

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',# 设置时间格式
    filename='useful_pred.log',  # 设置日志文件名
    filemode='a'
)

MODEL_NAME  = ['DualBertRationaleUsefulModel', 'TextUsefulModel']


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        初始化早停机制

        参数:
        patience: 在验证集上停止改进的步数
        min_delta: 可接受的损失改善的最小值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        调用函数用于判断是否早停

        参数:
        val_loss: 当前模型在验证集上的损失
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print("早停触发：验证损失在连续 {} 个时期未提高。".format(self.patience))

        return self.early_stop



class DualCrossAttnUsefulModel(nn.Module):
    def __init__(self,config):
        super(DualCrossAttnUsefulModel, self).__init__()
        self.bert_content = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.bert_rationale = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)

        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_rationale.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.cross_attn = DualCrossAttentionFeatureAggregator(4,config['emb_dim'])

        self.classifier = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                                        nn.ReLU(),
                                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                                        nn.Sigmoid()
                                        )


        def forward(self, **kwargs):
            rationales = torch.cat([kwargs['FTR_2'], kwargs['FTR_3']] , dim=0)
            rationale_masks = torch.cat([kwargs['FTR_2_masks'], kwargs['FTR_3_masks']], dim=0)

            content_features = self.bert_content(kwargs['content'],attention_mask=kwargs['content_masks'])[0]
            rationale_features = self.bert_rationale(
                rationales,
                attention_mask=rationale_masks
            )[0]




class DualBertRationaleUsefulModel(nn.Module):

    def __init__(self,config):
        super(DualBertRationaleUsefulModel, self).__init__()
        self.bert_content = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.bert_rationale = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)

        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_rationale.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.cross_attn = nn.MultiheadAttention(config['emb_dim'], 8,batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                                            nn.ReLU(),
                                            nn.Linear(config['model']['mlp']['dims'][-1], 1),
                                            nn.Sigmoid()
                                            )

    def forward(self, **kwargs):
       rationales = torch.cat([kwargs['FTR_2'], kwargs['FTR_3']] , dim=0)
       rationale_masks = torch.cat([kwargs['FTR_2_masks'], kwargs['FTR_3_masks']], dim=0)
       rationale_features = self.bert_rationale(
           rationales,
           attention_mask=rationale_masks
       )[0]
       content_features = self.bert_content(kwargs['content'],attention_mask=kwargs['content_masks'])[0]

       ca_out = self.cross_attn(
                   query=torch.cat([content_features, content_features.clone()], dim=0),
                   key=rationale_features,
                   value=rationale_features,
                   key_padding_mask=~rationale_masks.bool()
               )[0]

       ca_out = torch.mean(
           ca_out,
           dim=1
       )
       out = self.classifier(ca_out).squeeze(-1)
       return out





class TextUsefulModel(nn.Module):

    def __init__(self,config):
        super(TextUsefulModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.classifier = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                                                nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(config['model']['mlp']['dims'][-1], 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                                )

        # print(self.bert_model)
        for name, param in self.bert_model.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, **kwargs):
        inputs = {
            'input_ids': torch.cat([kwargs['FTR_2'], kwargs['FTR_3']], dim=0),
            'attention_mask': torch.cat([kwargs['FTR_2_masks'], kwargs['FTR_3_masks']], dim=0)
        }

        return self.classifier(torch.mean(self.bert_model(**inputs)[0], dim=1)).squeeze(-1)




def compute_metrics(predictions, labels):
    """
    计算二分类任务的各类指标

    参数:
    predictions: 模型预测的标签 (列表或 numpy 数组)
    labels: 真实标签 (列表或 numpy 数组)，
            0 表示真实 (real)，1 表示虚假 (fake)

    返回:
    dict: 包含 recall、precision 和 f1 score 的字典
    """
    # 计算各类指标
    loss_fn = nn.BCELoss()
    loss = loss_fn(predictions, labels.float()).item()
    predictions = predictions.numpy()
    labels = labels.numpy()
    predictions = (predictions > 0.5).astype(int)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # 计算针对 unuseful (label=0) 和 useful (label=1) 的 recall
    recall_unuseful = recall_score(labels, predictions, pos_label=0)
    recall_useful = recall_score(labels, predictions, pos_label=1)

    return {
        'loss':loss,
        'acc':accuracy_score(labels, predictions),
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'recall_unuseful': recall_unuseful,
        'recall_useful': recall_useful
    }


def load_model_if_exists(model, model_path):
    """
    判断模型文件是否存在，如果存在则加载参数。

    参数:
    model: 要加载参数的模型实例
    model_path: 模型参数文件的路径

    返回:
    bool: 如果成功加载模型参数返回 True，否则返回 False
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"模型参数已加载，路径：{model_path}")
        return True
    else:
        print(f"模型文件不存在，路径：{model_path}")
        return False

def test_model(model,data_loader):
    model.eval()
    test_labels = []
    pred = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            batch_data = data2gpu(batch, True, 'val')
            test_labels.append(torch.cat([batch_data['FTR_2_acc'], batch_data['FTR_3_acc']], dim=0))
            outputs = model(**batch_data)
            pred.append(outputs)

    test_labels = torch.cat(test_labels).cpu()
    pred = torch.cat(pred).cpu()
    metrics = compute_metrics(pred, test_labels)
    print(metrics)
    return metrics


def train_model(model,config,train_loader,val_loader):
    model.train()
    loss_fn = nn.BCELoss()
    early_stopping = EarlyStopping(patience=config['early_stop'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    for epoch in range(50):
        avg_loss = 0.0
        for batch_data in tqdm(train_loader):
            if config['use_cuda']:
                batch_data = data2gpu(batch_data, True, 'train')

            labels = torch.cat([batch_data['FTR_2_acc'], batch_data['FTR_3_acc']], dim=0).to(dtype=torch.float32)
            outputs = model(**batch_data)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                avg_loss += loss.item()
        avg_loss /= len(train_loader)
        print(f'epoch {epoch}, avg_loss: {avg_loss}')
        print('start val........')
        metrics = test_model(model,val_loader)
        if early_stopping(metrics['loss']):
            print(f'early stopping in epoch {epoch}')
            break



if __name__ == '__main__':
    print(os.getcwd())
    config = json.loads(open('../config/arg_config_win.json', 'r').read())
    train_loader, val_loader, test_loader = load_data(config)
    model_name = 'DualBertRationaleUsefulModel'
    if model_name == 'TextUsefulModel':
        model = TextUsefulModel(config)
        model_path = 'text_useful_model.pth'
    else:
        model = DualBertRationaleUsefulModel(config)
        model_path = 'dual_bert_rationale_useful_model.pth'


    if config['use_cuda']:
        devices = try_all_gpus()
        if len(devices) > 1:
            print('use multiple gpus')
            model = torch.nn.DataParallel(model)

        model = model.cuda()
    if load_model_if_exists(model, model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        train_model(model,config,train_loader,val_loader)
        torch.save(model.state_dict(), model_path)

    print('start test........')
    test_metrics = test_model(model,test_loader)

    logging.info(f"test_metrics: {test_metrics}")





