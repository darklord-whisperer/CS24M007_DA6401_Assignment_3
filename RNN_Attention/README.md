# RNN with Attention

## Description

This project implements an attention-based sequence-to-sequence model for transliteration (Hindi to Latin script). It uses recurrent neural networks (RNNs) with an encoder-decoder architecture and a Bahdanau-style attention mechanism to align input and output sequences. By attending to different parts of the input sequence during decoding, the model learns to transliterate characters more accurately.

## Usage

1. **Environment Setup:** Install required Python libraries, such as PyTorch and Weights & Biases (wandb). For example: `pip install torch wandb`  
2. **Prepare Data:** The model uses the Dakshina Hindi transliteration dataset. By default, it expects the training, development, and test sets in TSV format at:  
   - `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv`  
   - `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv`  
   - `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`  
3. **Training:** Run the training script with the desired arguments. For example: `python train_attention.py --train_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv --dev_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv --test_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`. You can adjust other hyperparameters via command-line arguments (see the *Arguments* section below).  
4. **Output:** The script will train the model and log training progress (loss, accuracy, etc.) to the console and to Weights & Biases (if enabled). After training, it will evaluate the model on the test set and can save the trained model to disk.  

## Weights & Biases Sweeps

A sweep configuration is provided to automate hyperparameter tuning with Weights & Biases. To run a sweep:  
1. Define or review the sweep configuration file (e.g., `attention_sweep.yaml`).  
2. Execute `wandb sweep attention_sweep.yaml` to create a new sweep. Copy the generated Sweep ID.  
3. Run `wandb agent <sweep_id>` to launch agents that perform multiple training runs with different hyperparameters.  

Metrics and models for each run will be logged to your W&B project.

## Attention Visualization

After training the model, you can visualize the attention alignments to understand how the model maps input to output. The model records attention weights for sample transliterations. Use a plotting library (such as Matplotlib or Seaborn) to create a heatmap of these attention weights, which shows how each output character attends to input characters. (For example, the provided notebook or script may include utilities to plot these attention maps.)

## Arguments

| Argument           | Type   | Default                                                  |
|--------------------|--------|----------------------------------------------------------|
| --train_path       | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv` |
| --dev_path         | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv`   |
| --test_path        | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`  |
| --batch_size       | int    | 256                                                      |
| --optimizer        | str    | `adam`                                                   |
| --num_epochs       | int    | 15                                                       |
| --num_enc_layers   | int    | 1                                                        |
| --num_dec_layers   | int    | 1                                                        |
| --num_hidden_layers| int    | 256                                                      |
| --embedding_size   | int    | 512                                                      |
| --dropout          | float  | 0.3                                                      |
| --cell             | str    | `lstm`                                                   |
