# Vanilla RNN

## Description

This directory contains a simple sequence-to-sequence RNN model (without attention) for transliteration (Hindi to Latin script). The architecture is similar to the attention model but uses only stacked RNN (LSTM) layers. This simpler model demonstrates the basic seq2seq setup without an attention mechanism.

## Usage

1. **Environment Setup:** Install required libraries (e.g., PyTorch and wandb). For example: `pip install torch wandb`  
2. **Prepare Data:** As with the attention model, use the Dakshina Hindi transliteration dataset. Ensure that the train/dev/test TSV files are available at the paths specified by the arguments.  
3. **Training:** Run the training script for the vanilla RNN. For example: `python train_vanilla.py --train_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv --dev_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv --test_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`. You can adjust hyperparameters via command-line arguments (see the *Arguments* section below).  
4. **Output:** The script will train the model and log metrics to the console and to Weights & Biases (if enabled). After training, it will evaluate on the test set and save the model if specified.  

## Generating Predictions

After training the vanilla RNN model, you can generate transliteration predictions for the test set or new inputs. A sample script or function (e.g., `predict_vanilla.py`) can be used to load the trained model and produce output. For example: `python predict_vanilla.py --model_path path/to/saved_model.pth --test_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`. This will output the transliterated results, which you can then compare with references or save to a file.

## Weights & Biases Sweep

A sweep configuration file (e.g., `vanilla_sweep.yaml`) is provided for hyperparameter tuning. To run it:  
1. Execute `wandb sweep vanilla_sweep.yaml` to create the sweep and note the Sweep ID.  
2. Run `wandb agent <sweep_id>` to launch training runs with different hyperparameters.  

Metrics from each run will be logged to W&B.

## Arguments

| Argument           | Type   | Default                                                  |
|--------------------|--------|----------------------------------------------------------|
| --train_path       | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv` |
| --dev_path         | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv`   |
| --test_path        | str    | `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`  |
| --batch_size       | int    | 64                                                       |
| --optimizer        | str    | `adam`                                                   |
| --num_epochs       | int    | 5                                                        |
| --num_enc_layers   | int    | 5                                                        |
| --num_dec_layers   | int    | 5                                                        |
| --num_hidden_layers| int    | 5                                                        |
| --embedding_size   | int    | 128                                                      |
| --dropout          | float  | 0                                                        |
| --cell             | str    | `lstm`                                                   |
| --sweep            | flag   | (optional)                                               |
