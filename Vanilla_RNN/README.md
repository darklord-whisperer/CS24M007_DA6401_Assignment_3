# Vanilla RNN

## Description

This directory contains a simple sequence-to-sequence RNN model (without attention) for transliteration (Hindi to Latin script). The architecture is similar to the attention model but uses only stacked RNN (LSTM) layers. This simpler model demonstrates the basic seq2seq setup without an attention mechanism.

## Usage

1. **Environment Setup:** Install required libraries (e.g., PyTorch and wandb). For example: `pip install torch wandb`  
2. **Prepare Data:** As with the attention model, use the Dakshina Hindi transliteration dataset. Ensure that the train/dev/test TSV files are available at the paths specified by the arguments.  
3. **Training:** Run the training script for the vanilla RNN. For example: `python train_vanilla.py --train_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv --dev_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv --test_path dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`. You can adjust hyperparameters via command-line arguments (see the *Arguments* section below).  
4. **Output:** The script will train the model and log metrics to the console and to Weights & Biases (if enabled). After training, it will evaluate on the test set and save the model if specified.  

### How to pass the hyperparameters configuration as command-line arguments to main.py  ?
```
--train_path # Path of the train data directory.
--dev_path # Path of the Validation data directory..
--test_path' #Path of the test data directory
--batch_size #Batch size
--optimizer #Optimizer
--num_epochs #Number of Epochs to train for.
--num_enc_layers #Number of Encoder Layers
--num_dec_layers #Number of Decoder Layers
--num_hidden_layers #Number of Hidden Layers
--embedding_size #Size of the embedding layer to be used
--dropout #Dropout Rate, 0 indicates no dropout.
--cell #Cell type
```


Example :-
python main.py --num_enc_layers 5  --num_dec_layers 5  --num_hidden_layers 128 \
--embedding_size 512 --cell 'lstm' --batch_size 64 --num_epochs 20 \
--optimizer 'adam' --dropout 0.2 \

### To generate a CSV containing predictions of the model :-
1. Run preprocessing.
2. Run :-
- Build the model
- Train the model
- And then use
- rnn_model.pred_2_csv(encoder_test_english, decoder_test_english, decoder_test_indic)

### Use WANDB sweep :-
In wandb_sweep_vanilla.py

- First Set your project name and username
wandb.init(project=project_name, entity=entity)

- To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
wandb.agent(sweep_id, project=project_name, function=train_wandb)

- Change the config_sweep as per the need, to sweep using different strategy.

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
