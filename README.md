# README

本仓库为["**Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**"](https://arxiv.org/abs/2309.12247)的实现，代码来源于https://github.com/ICTMCG/ARG

- ## Dataset

  - GPT数据集（原始论文数据集）
    - 原作者提供的数据集可通过 ["Application to Use the Datasets from ARG for Fake News Detection"](https://forms.office.com/r/DfVwbsbVyM) 获取
  - Qwen数据集
    - 通过Qwen生成的Rationale数据集可以通过https://github.com/LYQ1-ai/QwenVLRationaleGenerate自行生成

- ## Code

  - python==3.10.13

  - CUDA: 11.3

  - 预训练模型

    - [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 、 [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)  、[swinv2-tiny][https://huggingface.co/microsoft/swinv2-tiny-patch4-window16-256] 

  - 通过Conda创建环境

    ```shell
    conda env create -f env.yaml
    ```

- ## Run

  - ARG Qwen Gossipcop

    ```sh
    python main.py --gpu 0 --seed 3759 --lr 5e-5 --model_name ARG --language en --root_path /home/lyq/DataSet/FakeNews/gossipcop --bert_path /home/lyq/Model/bert-base-uncased --data_name en-arg --data_type rationale --rationale_usefulness_evaluator_weight 1.5 --llm_judgment_predictor_weight 1.0 --save_log_dir /logs/json/arg_qwen_gossipcop --dataset arg_qwen_gossipcop --batchsize 64 --max_len 170
    ```

    - 需要替换的参数
      - bert_path: Bert所在文件夹 
      - root_path：指定数据集根目录

  - ARG VL Qwen Gossipcop

    ```shell
    python main.py --gpu 0 --seed 3759 --lr 5e-5 --model_name ARG_VL --language en --root_path /home/lyq/DataSet/FakeNews/gossipcop --bert_path /home/lyq/Model/bert-base-uncased --data_name en-arg --data_type rationale --rationale_usefulness_evaluator_weight 1.5 --llm_judgment_predictor_weight 1.0 --save_log_dir /logs/json/arg_vl_qwen_gossipcop --dataset arg_qwen_gossipcop --image_encoder_path /home/lyq/Model/swinv2-tiny-patch4-window16-256 --batchsize 64 --max_len 170
    ```

    - 需要替换的参数
      - bert_path: Bert所在文件夹 
      - root_path：指定数据集根目录
      - image_encoder_path: 指定图像预处理模型(如swinv2-tiny)路径

  

