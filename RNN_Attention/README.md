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


### Use WANDB sweep :-
In wandb_sweep_attention.py

- First Set your project name and username
wandb.init(project=project_name, entity=entity)

- To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
wandb.agent(sweep_id, project=project_name, function=train_wandb)

- Change the config_sweep as per the need, to sweep using different strategy.


### To visualize attention :- 
In attention_visualization.py set the ids of the preprocessed images which you want to visualize for.
By default it will pick random images from test set.

### To check word level accuracy and get predictions :- 
- Run main.py in attention
- It will decode each sequence from the test set and calculate accuracy on the test set and save the predictions as Predictions_Vanilla.csv

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
