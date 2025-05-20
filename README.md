# DA6401 : Fundamentals of Deep Learning - Assignment 3

This repository contains the code and notebooks for Assignment 3 of the DA6401 course (Fundamentals of Deep Learning). The assignment consists of three parts:

- **GPT2 Finetuning**: Fine-tune GPT-2 on a lyrics dataset to generate lyrics that start with the prefix "I love deep learning."
- **RNN with Attention**: Build an RNN-based sequence-to-sequence model with attention for transliteration (Hindi to Latin script).
- **Vanilla RNN**: Implement a simpler RNN-based transliteration model without attention.

## Table of Contents

- [GPT2 Finetuning](GPT2_finetune/README.md) – Fine-tuning GPT-2 on a lyrics dataset.  
- [RNN with Attention](RNN_Attention/README.md) – Attention-based transliteration using RNNs.  
- [Vanilla RNN](Vanilla_RNN/README.md) – Simple transliteration model with a vanilla RNN.  
- [Report](#report) – Interactive project report on Weights & Biases.  
- [📂 Dataset Setup](#-dataset-setup)  
- [Results](#results) – Test‐set word‐level accuracies for both models.

---

## Report

Interactive project dashboard and logs are available on Weights & Biases:  
[W&B Assignment 3 Report](https://wandb.ai/cs24m007-iit-madras/Alik_CS24M007_DA6401_DeepLearning_Assignment-3/reports/Assignment-3-RNNs--VmlldzoxMjc4ODA5Nw)

## 📂 Dataset Setup

1. **Download the dataset**  
   Clone or download the Dakshina transliteration dataset from:  
   `https://github.com/google-research-datasets/dakshina`

2. **Extract the archive**  
   Ensure you have the folder structure:
   dakshina_dataset_v1.0/
└── hi/
└── lexicons/
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv

3. **Set dataset paths**  
- **Google Colab**  
  Mount your Drive and point to the TSV files, for example:  
  ```bash
  --train_path "/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
  --dev_path   "/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
  --test_path  "/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
  ```
- **Local machine**  
  If you cloned the repo and the dataset resides alongside it, use:  
  ```bash
  --train_path "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
  --dev_path   "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
  --test_path  "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
  ```

---

## Results

- **Vanilla RNN** test word-level accuracy: 36.38%  
- **Attention RNN** test word-level accuracy: 39.16%

Corresponding prediction CSV files are saved in their respective subfolders.
