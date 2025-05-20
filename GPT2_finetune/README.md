# GPT2 Finetuning

## Description

In this notebook, we fine-tune the GPT-2 language model on a dataset of song lyrics. GPT-2 is a transformer-based autoregressive language model developed by OpenAI that generates coherent text given a prompt. The goal is to train GPT-2 to generate lyrics that continue from the prefix "I love deep learning."

## Running the Notebook

1. **Environment Setup:** Ensure you have Python (e.g., 3.6+) and install necessary libraries. You will need PyTorch and the Hugging Face Transformers library. For example: `pip install torch transformers`  
2. **Open the Notebook:** Navigate to the `GPT2_finetune` folder and open the `GPT2_finetune.ipynb` notebook in Jupyter or Colab.  
3. **Execute Cells:** Run all cells in sequence. The notebook will load the lyrics dataset, fine-tune the GPT-2 model, and generate lyrics based on the given prefix.  
4. **View Output:** After training, the notebook will output generated lyrics in the cell outputs. You can adjust hyperparameters (such as number of epochs or learning rate) and re-run cells to experiment with different results.  
5. **Saving the Model (Optional):** The last two cells of the notebook save the fine-tuned model and tokenizer to disk. These steps are optional; you can skip them if you do not need to save the model locally.  
