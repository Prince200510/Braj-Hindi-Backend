import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from model import create_model, create_optimizer, masked_loss, masked_accuracy

class BrajHindiDataProcessor:
    def __init__(self, max_seq_len: int = 25, vocab_size: int = 10000):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.braj_tokenizer = None
        self.hindi_tokenizer = None
        
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def load_and_preprocess_data(self, csv_path: str) -> Tuple[List[str], List[str]]:
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        hindi_texts = df['hindi_sentence'].astype(str).tolist()
        braj_texts = df['braj_translation'].astype(str).tolist()
        braj_texts = [self.preprocess_text(text) for text in braj_texts]
        hindi_texts = [self.preprocess_text(text) for text in hindi_texts]
        
        valid_pairs = [(b, h) for b, h in zip(braj_texts, hindi_texts) if b and h and b != 'nan' and h != 'nan']
        braj_texts, hindi_texts = zip(*valid_pairs) if valid_pairs else ([], [])
        
        print(f"Loaded {len(braj_texts)} text pairs")
        return list(braj_texts), list(hindi_texts)
    
    def train_tokenizers(self, braj_texts: List[str], hindi_texts: List[str]):
        print("Training SentencePiece tokenizers hai bhai ")
        
        braj_text_file = "temp_braj.txt"
        hindi_text_file = "temp_hindi.txt"
        
        with open(braj_text_file, 'w', encoding='utf-8') as f:
            for text in braj_texts:
                f.write(text + '\n')
        
        with open(hindi_text_file, 'w', encoding='utf-8') as f:
            for text in hindi_texts:
                f.write(text + '\n')
        
        spm.SentencePieceTrainer.train(
            input=braj_text_file,
            model_prefix='braj_tokenizer',
            vocab_size=self.vocab_size,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            character_coverage=0.9995,
            num_threads=4,
            split_by_unicode_script=True,
            split_by_number=True,
            split_by_whitespace=True,
            treat_whitespace_as_suffix=False,
            allow_whitespace_only_pieces=False,
            split_digits=True,
            user_defined_symbols=['<START>', '<END>']
        )
        
        spm.SentencePieceTrainer.train(
            input=hindi_text_file,
            model_prefix='hindi_tokenizer',
            vocab_size=self.vocab_size,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            character_coverage=0.9995,
            num_threads=4,
            split_by_unicode_script=True,
            split_by_number=True,
            split_by_whitespace=True,
            treat_whitespace_as_suffix=False,
            allow_whitespace_only_pieces=False,
            split_digits=True,
            user_defined_symbols=['<START>', '<END>']
        )
    
        self.braj_tokenizer = spm.SentencePieceProcessor(model_file='braj_tokenizer.model')
        self.hindi_tokenizer = spm.SentencePieceProcessor(model_file='hindi_tokenizer.model')
        
        os.remove(braj_text_file)
        os.remove(hindi_text_file)
        
        print(f"Braj vocabulary size: {self.braj_tokenizer.get_piece_size()}")
        print(f"Hindi vocabulary size: {self.hindi_tokenizer.get_piece_size()}")
    
    def tokenize_and_pad(self, texts: List[str], tokenizer: spm.SentencePieceProcessor, add_start_end: bool = False) -> np.ndarray:
        tokenized = []
        
        for text in texts:
            tokens = tokenizer.encode(text, out_type=int)
            
            if add_start_end:
                tokens = [2] + tokens + [3]  
            
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            else:
                tokens = tokens + [0] * (self.max_seq_len - len(tokens))  
            tokenized.append(tokens)
        return np.array(tokenized, dtype=np.int32)
    
    def prepare_dataset(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        braj_texts, hindi_texts = self.load_and_preprocess_data(csv_path)
        self.train_tokenizers(braj_texts, hindi_texts)
        print("Tokenizing sequences...")
        braj_sequences = self.tokenize_and_pad(braj_texts, self.braj_tokenizer)
        hindi_sequences = self.tokenize_and_pad(hindi_texts, self.hindi_tokenizer, add_start_end=True)
        decoder_input = hindi_sequences[:, :-1]  
        decoder_target = hindi_sequences[:, 1:]  
        
        print(f"Dataset shape: Braj={braj_sequences.shape}, Hindi_input={decoder_input.shape}, Hindi_target={decoder_target.shape}")
        
        return braj_sequences, decoder_input, decoder_target, (braj_texts, hindi_texts)


def calculate_bleu_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    def bleu_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        return overlap / sum(pred_ngrams.values()) if sum(pred_ngrams.values()) > 0 else 0.0
    
    bleu_scores = {f'bleu_{i}': 0.0 for i in range(1, 4)}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        for n in range(1, 4):
            bleu_scores[f'bleu_{n}'] += bleu_n(pred_tokens, ref_tokens, n)
    num_sentences = len(predictions)
    for key in bleu_scores:
        bleu_scores[key] /= num_sentences if num_sentences > 0 else 1
    
    return bleu_scores


class TranslationTrainer:
    def __init__(self, model, optimizer, data_processor):
        self.model = model
        self.optimizer = optimizer
        self.data_processor = data_processor
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        
    @tf.function
    def train_step(self, inp, tar_inp, tar_real):
        with tf.GradientTape() as tape:
            predictions = self.model([inp, tar_inp], training=True)
            loss = masked_loss(tar_real, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(masked_accuracy(tar_real, predictions))
    
    def evaluate_model(self, val_data: Tuple, original_texts: Tuple, sample_size: int = 50) -> Dict[str, float]:
        val_braj, val_hindi_inp, val_hindi_tar = val_data
        original_braj, original_hindi = original_texts
        indices = np.random.choice(len(val_braj), min(sample_size, len(val_braj)), replace=False)
        
        predictions = []
        references = []
        
        for idx in indices:
            braj_text = original_braj[idx]
            hindi_text = original_hindi[idx]
            translated = self.translate_text(braj_text)
            predictions.append(translated)
            references.append(hindi_text)
        bleu_scores = calculate_bleu_scores(predictions, references)
        return bleu_scores
    
    def translate_text(self, braj_text: str) -> str:
        braj_tokens = self.data_processor.braj_tokenizer.encode(braj_text, out_type=int)
        
        if len(braj_tokens) > self.data_processor.max_seq_len:
            braj_tokens = braj_tokens[:self.data_processor.max_seq_len]
        else:
            braj_tokens = braj_tokens + [0] * (self.data_processor.max_seq_len - len(braj_tokens))
        
        encoder_input = tf.expand_dims(braj_tokens, 0)
        decoder_input = [2] 
        output = tf.expand_dims(decoder_input, 0)
        
        for _ in range(self.data_processor.max_seq_len):
            dec_inp = output.numpy()[0]
            if len(dec_inp) < self.data_processor.max_seq_len:
                dec_inp = list(dec_inp) + [0] * (self.data_processor.max_seq_len - len(dec_inp))
            else:
                dec_inp = dec_inp[:self.data_processor.max_seq_len]
            dec_inp = tf.expand_dims(dec_inp, 0)
            predictions = self.model([encoder_input, dec_inp], training=False)
            predicted_id = tf.cast(tf.argmax(predictions[:, -1:, :], axis=-1), tf.int32)

            if predicted_id == 3:  
                break
            output = tf.concat([output, predicted_id], axis=-1)
        output_tokens = output.numpy()[0][1:]  
        output_tokens = [int(token) for token in output_tokens if token not in [0, 3]]  
        
        translated_text = self.data_processor.hindi_tokenizer.decode(output_tokens)
        return translated_text
    
    def train(self, train_data: Tuple, val_data: Tuple, original_texts: Tuple, 
              epochs: int = 50, batch_size: int = 64):
        """Train the model"""
        train_braj, train_hindi_inp, train_hindi_tar = train_data
        dataset = tf.data.Dataset.from_tensor_slices({
            'braj': train_braj,
            'hindi_inp': train_hindi_inp,
            'hindi_tar': train_hindi_tar
        })
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        best_bleu = 0.0
        history = {'loss': [], 'accuracy': [], 'bleu_1': [], 'bleu_2': [], 'bleu_3': []}
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            
            progress_bar = tqdm(dataset, desc='Training')
            for batch in progress_bar:
                self.train_step(batch['braj'], batch['hindi_inp'], batch['hindi_tar'])
                
                progress_bar.set_postfix({
                    'loss': f'{self.train_loss.result():.4f}',
                    'accuracy': f'{self.train_accuracy.result():.4f}'
                })
            
            if (epoch + 1) % 5 == 0:  
                print("Evaluating model...")
                bleu_scores = self.evaluate_model(val_data, original_texts)
                
                print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}, "
                      f"BLEU-2: {bleu_scores['bleu_2']:.4f}, "
                      f"BLEU-3: {bleu_scores['bleu_3']:.4f}")
                
                if bleu_scores['bleu_1'] > best_bleu:
                    best_bleu = bleu_scores['bleu_1']
                    self.model.save_weights('best_model_weights.weights.h5')
                    print(f"New best model saved! BLEU-1: {best_bleu:.4f}")
                
                history['bleu_1'].append(bleu_scores['bleu_1'])
                history['bleu_2'].append(bleu_scores['bleu_2'])
                history['bleu_3'].append(bleu_scores['bleu_3'])
            
            history['loss'].append(float(self.train_loss.result()))
            history['accuracy'].append(float(self.train_accuracy.result()))
            
            print(f'Loss: {self.train_loss.result():.4f}, Accuracy: {self.train_accuracy.result():.4f}')
        
        return history


def plot_training_history(history: Dict):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(history['accuracy'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    if history['bleu_1']:
        epochs_eval = list(range(4, len(history['loss']), 5))  # Every 5 epochs
        ax3.plot(epochs_eval, history['bleu_1'], label='BLEU-1', marker='o')
        ax3.plot(epochs_eval, history['bleu_2'], label='BLEU-2', marker='s')
        ax3.plot(epochs_eval, history['bleu_3'], label='BLEU-3', marker='^')
        ax3.set_title('BLEU Scores')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('BLEU Score')
        ax3.legend()
        ax3.grid(True)
    
    ax4.text(0.1, 0.9, 'Training completed!', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Final Loss: {history["loss"][-1]:.4f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Final Accuracy: {history["accuracy"][-1]:.4f}', fontsize=12, transform=ax4.transAxes)
    if history['bleu_1']:
        ax4.text(0.1, 0.3, f'Best BLEU-1: {max(history["bleu_1"]):.4f}', fontsize=12, transform=ax4.transAxes)
    ax4.set_title('Training Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function"""
    print("Starting Braj-to-Hindi Neural Machine Translation Training")
    print("=" * 60)
    
    CSV_PATH = "Dataset_v1.csv"
    MAX_SEQ_LEN = 25
    VOCAB_SIZE = 10000  
    EPOCHS = 50
    BATCH_SIZE = 32
    
    data_processor = BrajHindiDataProcessor(MAX_SEQ_LEN, VOCAB_SIZE)
    braj_sequences, hindi_input, hindi_target, original_texts = data_processor.prepare_dataset(CSV_PATH)
    train_braj, val_braj, train_hindi_inp, val_hindi_inp, train_hindi_tar, val_hindi_tar = train_test_split(
        braj_sequences, hindi_input, hindi_target, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(train_braj)}")
    print(f"Validation set size: {len(val_braj)}")
    
    input_vocab_size = data_processor.braj_tokenizer.get_piece_size()
    target_vocab_size = data_processor.hindi_tokenizer.get_piece_size()
    
    model = create_model(input_vocab_size, target_vocab_size, MAX_SEQ_LEN)
    optimizer = create_optimizer(d_model=512)
    
    print(f"Model created with {input_vocab_size} input vocab and {target_vocab_size} target vocab")
    
    trainer = TranslationTrainer(model, optimizer, data_processor)
    
    train_data = (train_braj, train_hindi_inp, train_hindi_tar)
    val_data = (val_braj, val_hindi_inp, val_hindi_tar)
    
    history = trainer.train(train_data, val_data, original_texts, EPOCHS, BATCH_SIZE)
    model.save_weights('final_model_weights.weights.h5')
    
    metadata = {
        'max_seq_len': MAX_SEQ_LEN,
        'vocab_size': VOCAB_SIZE,
        'input_vocab_size': input_vocab_size,
        'target_vocab_size': target_vocab_size,
        'original_texts': original_texts    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nTraining completed!")
    print("Saved files:")
    print("  - best_model_weights.weights.h5 (best model)")
    print("  - final_model_weights.weights.h5 (final model)")
    print("  - braj_tokenizer.model (Braj tokenizer)")
    print("  - hindi_tokenizer.model (Hindi tokenizer)")
    print("  - model_metadata.pkl (training metadata)")
    print("  - training_history.png (training plots)")
    
    plot_training_history(history)
    
    print("\n🔍 Sample translations:")
    test_texts = ["हौं", "तुम", "करब", "जाब"]
    for text in test_texts:
        if text in [pair[1] for pair in zip(*original_texts)]:  
            translated = trainer.translate_text(text)
            print(f"  Braj: '{text}' → Hindi: '{translated}'")


if __name__ == "__main__":
    main()
