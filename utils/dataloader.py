import os
import pickle
import warnings

from PIL import Image

from torch.utils.data.dataset import Dataset


warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

import pandas as pd
import json

from transformers import BertTokenizer, AutoImageProcessor, AutoTokenizer, BeitFeatureExtractor, ViTImageProcessor
from torch.utils.data import TensorDataset, DataLoader



label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2
}

label_str2int_dict = {
    'real': 0,
    'fake': 1,
    'other': 2,
}

def word2input(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def sent2tensor(text, max_len, tokenizer):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # 添加特殊标记 [CLS] 和 [SEP]
        max_length=max_len,       # 最大长度
        truncation=True,          # 超出部分截断
        padding='max_length',     # 填充到最大长度
        return_tensors="pt"       # 返回 PyTorch 的张量
    )
    input_ids = encoded['input_ids'].squeeze(0)  # 提取 input_ids 并去掉多余的维度
    attention_mask = encoded['attention_mask'].squeeze(0)  # 提取 attention_mask 并去掉多余的维度
    return input_ids, attention_mask

def process_images(image_paths,image_processor):
    if len(image_paths) == 0 or image_processor is None:
        return None
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    return image_processor(images = images,return_tensors = 'pt').pixel_values

def process_image(image_path,image_processor):
    i = Image.open(image_path).convert("RGB")
    return image_processor([i],return_tensors = 'pt').pixel_values.squeeze(0)



class ARGDataset(Dataset):
    def is_valid_item(self,item):
        """
        :param item: {
            "content":"",
            "label":"real",
            "time":1448118196000,
            "source_id":893,
            "td_rationale":"",
            "td_pred":"other",
            "td_acc":0,
            "cs_rationale":"无法确定。因为没有给出具体的消息内容，无法判断其真实性。",
            "cs_pred":"other",
            "cs_acc":0,
            "split":"train"
        }
        :return:
        """
        return all([
            item['content'] is not None and len(item['content']) > 0,
            item['td_rationale'] is not None and len(item['td_rationale']) > 0,
            item['cs_rationale'] is not None and len(item['cs_rationale']) > 0
        ])

    def __init__(self, df: pd.DataFrame, image_processor, max_len, tokenizer, cache_file="dataset_cache.pkl"):
        """
        :param df: {

        }
        :param image_processor:
        :param max_len:
        :param tokenizer:
        :param cache_file:
        """
        super().__init__()

        # 如果缓存文件存在，直接加载
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
            print(f"Loaded cached dataset from {cache_file}")
        else:
            # 数据预处理
            print(f'before filter:{df.shape[0]}')
            df = df[df.apply(lambda x: self.is_valid_item(x), axis=1)]
            print(f'after filter:{df.shape[0]}')
            df["source_id"] = df['source_id'].astype(dtype=str)
            df['content'], df['content_masks'] = zip(*df['content'].apply(lambda x: sent2tensor(x, max_len, tokenizer)))
            df['td_rationale'], df['td_mask'] = zip(*df['td_rationale'].apply(lambda x: sent2tensor(x, max_len, tokenizer)))
            df['cs_rationale'], df['cs_mask'] = zip(*df['cs_rationale'].apply(lambda x: sent2tensor(x, max_len, tokenizer)))
            if 'caption' in df.columns:
                df['caption'],df['caption_mask'] = zip(*df['caption'].apply(lambda x: sent2tensor(x, max_len, tokenizer)))

            if  image_processor is not None:
                df['image'] = df['image_path'].apply(lambda x: process_image(x, image_processor))

            # 可选列重命名
            df.rename(columns={
                'td_rationale': 'FTR_2',
                'td_mask': 'FTR_2_masks',
                'td_pred':'FTR_2_pred',
                'td_acc':'FTR_2_acc',
                'cs_rationale': 'FTR_3',
                'cs_mask': 'FTR_3_masks',
                'cs_pred':'FTR_3_pred',
                'cs_acc':'FTR_3_acc',
            }, inplace=True)

            # 转换为字典列表
            self.data = df.to_dict(orient='records')

            # 保存到缓存文件
            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)
            print(f"Processed and cached dataset to {cache_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_dataloader_qwen_goss(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path):
    df = pd.read_csv(f'{path}/{data_type}.csv', encoding='utf-8')
    image_dir = f'{path}/images'
    df['image_path'] = df['image_id'].apply(lambda x: f'{image_dir}/{x}_top_img.png')
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_path) if image_encoder_path is not None else None
    print(f'load {data_type} data: sum {df.shape[0]} real {(df["label"]==0).sum()} fake {(df["label"]==1).sum()}')
    cache_file = f'{path}/{data_type}.pkl'
    ds = ARGDataset(df,image_processor,max_len,tokenizer,cache_file)
    return DataLoader(
        ds,               # 传入自定义的 Dataset
        batch_size=batch_size, # 批量大小
        shuffle=True,          # 是否打乱数据
        num_workers=4,         # 数据加载线程数
        collate_fn=None        # 使用默认的合并函数
    )





def get_dataloader_qwen_twitter(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path):
    def get_image_path_dict():
        image_dir = f'{path}/images/'
        return { f.split('.')[0]: f'{image_dir}/{f}' for f in os.listdir(image_dir)}



    tokenizer = BertTokenizer.from_pretrained(bert_path)
    data_file_name = f'{path}/{data_type}.csv'
    df = pd.read_csv(data_file_name)
    image_dict = get_image_path_dict()
    df['image_path'] = df['image_id'].apply(lambda image_id: image_dict[image_id])
    print(f'load {data_type} data: sum {df.shape[0]} real {(df["label"]==1).sum()} fake {(df["label"]==0).sum()}')

    # df = balance_data(df,data_type)
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_path) if image_encoder_path is not None else None
    ds = ARGDataset(df, image_processor, max_len, tokenizer,cache_file=f'{path}/{data_type}.pkl')
    return DataLoader(
        ds,  # 传入自定义的 Dataset
        batch_size=batch_size,  # 批量大小
        shuffle=True,  # 是否打乱数据
        num_workers=4
    )

def get_dataloader_qwen_weibo(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path):

    def get_image_dict(root_path):
        image_dir_list = [f'{root_path}/nonrumor_images/', f'{root_path}/rumor_images/']
        image_dict = {}
        for image_dir in image_dir_list:
            image_dict.update({
                f.split('.')[0]: f'{image_dir}/{f}' for f in os.listdir(image_dir)
            })
        return image_dict

    df = pd.read_csv(f'{path}/{data_type}.csv', encoding='utf-8')
    image_dict = get_image_dict(path)

    df['image_path'] = df['image_id'].apply(lambda x: image_dict[x])
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_path) if image_encoder_path is not None else None
    cache_file = f'{path}/cache/{data_type}.pkl'
    ds = ARGDataset(df, image_processor, max_len, tokenizer, cache_file)
    return DataLoader(
        ds,  # 传入自定义的 Dataset
        batch_size=batch_size,  # 批量大小
        shuffle=shuffle,  # 是否打乱数据
        num_workers=4,  # 数据加载线程数
        collate_fn=None  # 使用默认的合并函数
    )

def get_dataloader_arg(path, max_len, batch_size, shuffle, bert_path, data_type, language):
    df = pd.read_json(f'{path}/{data_type}.json',encoding = 'utf-8')
    if language == 'ch':
        df['label'] = df['label'].apply(lambda x: label_dict_ftr_pred[x]).astype(int)
        df['td_pred'] = df['td_pred'].apply(lambda x: label_dict_ftr_pred[x]).astype(int)
        df['cs_pred'] = df['td_pred'].apply(lambda x: label_dict_ftr_pred[x]).astype(int)


    print(f'load {data_type} data: sum {df.shape[0]} real {(df["label"]==label_str2int_dict["real"]).sum()} fake {(df["label"]==label_str2int_dict["fake"]).sum()}')
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    cache_file = f'{path}/{data_type}.pkl'
    ds = ARGDataset(df,None,max_len,tokenizer,cache_file)
    return DataLoader(
        ds,  # 传入自定义的 Dataset
        batch_size=batch_size,  # 批量大小
        shuffle=True,  # 是否打乱数据
        collate_fn=None  # 使用默认的合并函数,
    )





def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language,dataset,image_encoder_path=None):
    """return :{
            'content': Tensor([batch_size,seq_len]),
            'contents_mask': Tensor([batch_size,seq_len]),
            'label': Tensor([batch_size]),
            'image_id': list[batch_size]
            'source_id': list[batch_size]
            'FTR_2': Tensor([batch_size,seq_len]),
            'FTR_2_pred': Tensor([batch_size,]),
            'FTR_2_acc': Tensor([batch_size,]),
            'FTR_2_mask': Tensor([batch_size,seq_len]),
            'image': Tensor([batch_size,3,224,224]),
            'caption': Tensor([batch_size,seq_len]),
            'caption_mask': Tensor([batch_size,seq_len]),
            ...
        }"""
    if dataset == 'arg_gpt_gossipcop':
        return get_dataloader_arg(path, max_len, batch_size, shuffle, bert_path, data_type, language)
    elif dataset == 'arg_qwen_gossipcop':
        return get_dataloader_qwen_goss(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path)
    elif dataset == 'arg_qwen_twitter':
        return get_dataloader_qwen_twitter(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path)
    elif dataset == 'arg_gpt_weibo':
        return get_dataloader_arg(path, max_len, batch_size, shuffle, bert_path, data_type, language)
    elif dataset == 'arg_qwen_weibo':
        return get_dataloader_qwen_weibo(path, max_len, batch_size, shuffle, bert_path, data_type, language,image_encoder_path)


def load_data(config):
    data_loaders = []
    for data_type in ['train', 'val', 'test']:
        data_loaders.append(
            get_dataloader(
            config['root_path'],
            config['max_len'],
            config['batchsize'],
            shuffle=True,
            bert_path=config['bert_path'],
            data_type=data_type,
            language=config['language'],
            dataset=config['dataset'],
            image_encoder_path=config['image_encoder_path'],)
        )
    return data_loaders