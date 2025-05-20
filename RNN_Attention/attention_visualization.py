import wandb
import numpy as np
import tensorflow as tf
import pandas as pd
from preprocess import *
from rnn_attention import *
import matplotlib.pyplot as plt
from matplotlib import font_manager
from seaborn import heatmap 
import random

# Load custom font for Indic characters
font_prop = font_manager.FontProperties(fname='./VesperLibre-Regular.ttf')

# W&B project settings
project_name = 'Alik_CS24M007_DA6401_DeepLearning_Assignment-3'
entity       = 'cs24m007-iit-madras'

# Safely configure GPU memory growth (or fall back to CPU)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Enabled memory growth for {len(physical_devices)} GPU(s).")
else:
    print("No GPU detected; running on CPU.")

wandb.init(project=project_name, entity=entity)

# File paths
train_path = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
dev_path   = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
test_path  = "/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"

# Preprocess data (returns TF Datasets and vocab/meta)
(
    (encoder_train_english, decoder_train_english, decoder_train_indic),
    (encoder_val_english, decoder_val_english, decoder_val_indic),
    (val_english, val_indic),
    (encoder_test_english, decoder_test_english, decoder_test_indic),
    (english_char_set, indic_char_set, max_seq_len_english_encoder, max_seq_len_indic_decoder),
    (indic_char_to_idx, indic_idx_to_char),
    (english_char_to_idx, english_idx_to_char)
) = preprocess(
    train_path, dev_path, test_path,
    batch_size=128
)

# Build the RNN+Attention model
rnn_model = Model(
    english_char_set, indic_char_set,
    max_seq_len_english_encoder, max_seq_len_indic_decoder,
    indic_char_to_idx, indic_idx_to_char,
    english_char_to_idx, english_idx_to_char,
    cell='gru',
    optimizer='adam',
    embedding_size=512,
    num_enc_layers=1,
    num_dec_layers=1,
    num_hidden_layers=256,
    dropout=0.2
)
rnn_model.build_model()

# Train
rnn_model.train(
    encoder_train_english, decoder_train_english, decoder_train_indic,
    encoder_val_english, decoder_val_english, decoder_val_indic,
    num_epochs=10, batch_size=128
)

# Pick some random samples for attention visualization
idx = random.sample(range(len(encoder_train_english)), 9)

# Run inference to collect predictions & attention weights
predictions = []
attentions  = []
test_df     = load_data(test_path)
test_indic  = test_df['indic'].values

rnn_model.inference_setup()
for i in idx:
    input_seq = encoder_test_english[i:i+1]
    output, attn_weights = rnn_model.decode_sequence(input_seq)
    print(test_indic[i].strip(), "→", output.strip())
    predictions.append(output)
    attentions.append(attn_weights)

# Prepare the attention heatmaps
test_english = test_df['english'].values
fig, ax = plt.subplots(3, 3, figsize=(23, 20))
ax = ax.flatten()

for plot_i, sample_i in enumerate(idx):
    output    = predictions[plot_i]
    attention = attentions[plot_i]
    inp_seq   = test_english[sample_i]
    ip_len    = len(inp_seq)
    op_len    = len(output)

    # Gather weights per decoder timestep
    weights = [attention[t][:ip_len] for t in range(op_len)]
    
    # Plot heatmap
    heatmap(
        weights, cbar=True, ax=ax[plot_i],
        cmap="Blues"
    )
    ax[plot_i].set_xticklabels(list(inp_seq))
    ax[plot_i].set_yticklabels(list(output), fontproperties=font_prop, rotation=0)
    ax[plot_i].set_title(f"Sample {sample_i}")

# Save and log to W&B
plt.tight_layout()
plt.savefig('./att_heatmap.jpg')
wandb.log({'Attention_Heatmap': wandb.Image(fig)})
plt.show()
