# ARG

Official repository for ["**Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**"](https://arxiv.org/abs/2309.12247), which has been accepted by AAAI 2024.

## Dataset

The experimental datasets where can be seen in `data` folder. Note that you can download the datasets only after an ["Application to Use the Datasets from ARG for Fake News Detection"](https://forms.office.com/r/DfVwbsbVyM) has been submitted.

## Code

### Requirements

- python==3.10.13

- CUDA: 11.3

- Python Packages:

  ```
  pip install -r requirements.txt
  ```

### Pretrained Models

You can download pretrained models ([bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)) and change paths (`bert_path`) in the corresponding scripts.

### Run

You can run this model through `run_zh.sh` for the Chinese dataset and `run_en.sh` for the English dataset. 

## How to Cite

```
@inproceedings{hu2024bad,
  title={{Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection}},
  author={Hu, Beizhe and Sheng, Qiang and Cao, Juan and Shi, Yuhui and Li, Yang and Wang, Danding and Qi, Peng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22105--22113},
  doi={10.1609/aaai.v38i20.30214},
  year={2024}
}
```

## Relevant Resources
- Paper List ``LLM-for-misinformation-research``: https://github.com/ICTMCG/LLM-for-misinformation-research/
- Tutorial @SIGIR 2024 ``Preventing and Detecting Misinformation Generated by Large Language Models``: https://sigir24-llm-misinformation.github.io/