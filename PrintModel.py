import json
import os
from collections import namedtuple

import torch
from tensorboardX import SummaryWriter
from torch import Tensor

from models.arg import ARGModel
from models.argVL import ARGVLModel
from pathlib import Path


MODEL_NAME = 'ARGVL'

MODEL_CONFIG_DICT = {
    'ARG' : 'config/arg_config.json',
    'ARGVL' : 'config/argvl_config.json',
}

MODEL_DICT = {
    'ARG' : ARGModel,
    'ARGVL' : ARGVLModel,
}



ARGInputSimple = namedtuple('ARGInputSimple', ['content', 'content_masks', 'FTR_2','FTR_3','FTR_2_masks','FTR_3_masks'])
ARGVLInputSimple = namedtuple('ARGVLInputSimple', ['content', 'content_masks', 'FTR_2','FTR_3','FTR_2_masks','FTR_3_masks','image'])

MODEL_INPUT_DICT = {
    'ARG': ARGInputSimple,
    'ARGVL': ARGVLInputSimple,
}


def load_config(model_name: str) -> dict:
    config_path = MODEL_CONFIG_DICT[model_name]
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def create_model(model_name: str, config: dict) -> torch.nn.Module:
    model_class = MODEL_DICT.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_class(config)


def prepare_input_sample(config: dict, model_name: str) -> namedtuple:
    input_sample = {
        'content': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
        'content_masks': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
        'FTR_2': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
        'FTR_3': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
        'FTR_2_masks': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
        'FTR_3_masks': torch.ones(config['batchsize'], config['max_len'], dtype=torch.int),
    }

    if model_name == 'ARGVL':
        input_sample['image'] = torch.rand(config['batchsize'], 3, 256, 256)

    return MODEL_INPUT_DICT[model_name](**input_sample)


if __name__ == '__main__':
    try:
        # Load configuration and create the model
        config = load_config(MODEL_NAME)
        model = create_model(MODEL_NAME, config)

        # Prepare an input sample for the model
        input_sample = prepare_input_sample(config, MODEL_NAME)
        traced_model = torch.jit.trace(model, input_sample, strict=False)
        # Set up a summary writer and add the graph to TensorBoard
        log_dir = './logs'  # Define a meaningful directory for logs
        with SummaryWriter(log_dir=log_dir) as sw:
            sw.add_graph(traced_model, input_sample)

        print(f"Model graph saved to TensorBoard logs in {log_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
