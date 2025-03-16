import sentencepiece as spm
import os
import logging

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(filename='../logs/text_feature_extraction.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    text_file = '../datasets/text_data.txt'
    model_prefix = 'sentencepiece'
    desired_vocab_size = 10000  # Original target

    # Temporary training to check max allowed vocab size
    spm.SentencePieceTrainer.Train(f'--input={text_file} --model_prefix=temp_model --vocab_size=1000')

    # Load the temporary model to check actual vocabulary size
    temp_model = spm.SentencePieceProcessor(model_file='temp_model.model')
    actual_vocab_size = len(temp_model)

    # Cleanup temporary model files
    os.remove('temp_model.model')
    os.remove('temp_model.vocab')

    # Adjust the vocab size if necessary
    if desired_vocab_size > actual_vocab_size:
        logging.warning(f"Desired vocab_size of {desired_vocab_size} is too high. Adjusting to {actual_vocab_size}.")
        desired_vocab_size = actual_vocab_size

    # Train the SentencePiece model with adjusted vocab size
    spm.SentencePieceTrainer.Train(f'--input={text_file} --model_prefix={model_prefix} --vocab_size={desired_vocab_size}')
    logging.info("SentencePiece model training completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")