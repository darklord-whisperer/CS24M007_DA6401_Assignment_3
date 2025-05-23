import tensorflow as tf
tf.config.run_functions_eagerly(True)
from preprocess import *
from rnn_attention import *
import wandb

def train_wandb(config = None):

    run = wandb.init(config=config, resume=True)

    cfg = wandb.config

    cell = cfg.cell
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    embedding_size = cfg.embedding_size
    num_enc_layers = 1
    num_dec_layers = 1
    num_hidden_layers = cfg.num_hidden_layers
    dropout = cfg.dropout
    optimizer = cfg.optimizer

    name = f"cell_{cfg.cell}_hidden_{cfg.num_hidden_layers}_encl_{num_enc_layers}_decl_{num_dec_layers}_emb_{cfg.embedding_size}_opt_{cfg.optimizer}_drop_{cfg.dropout}"

    train_path = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    test_path = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    wandb.run.name = name
    wandb.run.save()

  
    


    #Generate training, validation and test batches (along with paddings, encodings).
    (encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = batch_size)  

    #Create a RNN-model
    #Generate training, validation and test batches (along with paddings, encodings).
    (encoder_train_english, decoder_train_english, decoder_train_indic), (encoder_val_english, decoder_val_english, decoder_val_indic), (val_english, val_indic), (encoder_test_english, decoder_test_english, decoder_test_indic), (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder), (indic_char_to_idx, indic_idx_to_char), (english_char_to_idx, english_idx_to_char) = preprocess(train_path, dev_path, test_path, batch_size = batch_size)  

    #Create a RNN-model
    rnn_model =  Model(english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder, indic_char_to_idx, indic_idx_to_char, english_char_to_idx, english_idx_to_char, cell = cell, optimizer = optimizer, embedding_size = embedding_size, num_enc_layers = num_enc_layers, num_dec_layers = num_dec_layers, num_hidden_layers = num_hidden_layers, dropout = dropout)
    rnn_model.build_model()

    rnn_model.train(encoder_train_english, decoder_train_english, decoder_train_indic, encoder_val_english, decoder_val_english, decoder_val_indic, num_epochs = num_epochs, batch_size = batch_size)
    
    '''
    val_df = load_data(test_path)
    val_indic = val_df['indic'].values
    ctr = 0

    rnn_model.inference_setup()

    
    for i in range(len(val_indic)):
        input_seq = encoder_test_english[i:i+1]
        output, _ = rnn_model.decode_sequence(input_seq)
        if val_indic[i].strip() == output.strip():
            ctr+=1
    word_level_val_acc = (ctr/len(val_indic))

    wandb.log({'word_level_val_acc': word_level_val_acc})
    '''

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
              },
    'early_terminate' : {
      'type': 'hyperband',
      'min_iter': 5
    },
    "parameters": {
        "embedding_size": {
            "values": [64,128,256,512]
        },
        "num_hidden_layers": {
            "values": [32, 64, 256, 512]
        },
        "cell": {
            "values": ["rnn", "lstm", "gru"]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "num_epochs": {
            "values": [10, 15]
        },
        "dropout": {
            "values": [0,0.2, 0.3]
        },
        "optimizer": {
            "values": ["adam", "nadam"]
        },
    }
}

project_name = "Alik_CS24M007_DA6401_DeepLearning_Assignment-3" #Add project name here
entity = "cs24m007-iit-madras" #Add username here

wandb.init(project=project_name, entity=entity)

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

wandb.agent(sweep_id, project=project_name, function=train_wandb)
