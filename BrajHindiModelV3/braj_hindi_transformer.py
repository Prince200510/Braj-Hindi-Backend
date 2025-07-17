import os
import json
import csv
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

import sentencepiece as spm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

warnings.filterwarnings('ignore')

def setup_colab_environment():
    if os.path.exists('/content'):
        print("Google Colab environment detected!")
        print("Setting up environment...")
        
        if os.path.exists('/content/drive/MyDrive'):
            print("Google Drive is mounted")
            
            search_paths = [
                "/content/drive/MyDrive/Colab Notebooks/BrajHindiModelV3/",
                "/content/drive/MyDrive/Colab Notebooks/Model13/BrajHindiModelV3/",
                "/content/drive/MyDrive/Model13/BrajHindiModelV3/",
                "/content/drive/MyDrive/BrajHindiModelV3/",
                "/content/drive/MyDrive/Model13/",
                "/content/drive/MyDrive/"
            ]
            
            found_files = []
            for search_path in search_paths:
                if os.path.exists(search_path):
                    files_in_path = []
                    if os.path.exists(os.path.join(search_path, "word.csv")):
                        files_in_path.append("word.csv")
                    if os.path.exists(os.path.join(search_path, "Dataset_v1.csv")):
                        files_in_path.append("Dataset_v1.csv")
                    
                    if files_in_path:
                        found_files.append((search_path, files_in_path))
                        print(f"Found in {search_path}: {', '.join(files_in_path)}")
            
            if not found_files:
                print("No data files found in Google Drive!")
                print("Please upload your files to one of these locations:")
                for path in search_paths[:3]:  
                    print(f"   - {path}")
                print("\n🔧 Or copy files to /content/ directory:")
                print("   !cp '/content/drive/MyDrive/path/to/your/word.csv' /content/")
                print("   !cp '/content/drive/MyDrive/path/to/your/Dataset_v1.csv' /content/")
            
        else:
            print("Google Drive not mounted!")
            print("Please mount Google Drive first:")
            print("from google.colab import drive")
            print("drive.mount('/content/drive')")
            print("\nAlternative: Upload files directly to /content/")
            print("Use the file upload button in Colab's file panel")
        
        content_files = []
        if os.path.exists('/content/word.csv'):
            content_files.append('word.csv')
        if os.path.exists('/content/Dataset_v1.csv'):
            content_files.append('Dataset_v1.csv')
        
        if content_files:
            print(f"Found in /content/: {', '.join(content_files)}")
        
        try:
            import sentencepiece
        except ImportError:
            print("Installing SentencePiece...")
            os.system('pip install sentencepiece')
        try:
            import sklearn
        except ImportError:
            print("Installing scikit-learn...")
            os.system('pip install scikit-learn')
        
        print("Environment setup complete!")
        return True
    return False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Config:
    @staticmethod
    def detect_environment():
        possible_drive_paths = [
            "/content/drive/MyDrive/Colab Notebooks/BrajHindiModelV3/",
            "/content/drive/MyDrive/Colab Notebooks/Model13/BrajHindiModelV3/",
            "/content/drive/MyDrive/Model13/BrajHindiModelV3/",
            "/content/drive/MyDrive/BrajHindiModelV3/",
            "/content/drive/MyDrive/Model13/",
            "/content/drive/MyDrive/"
        ]
        
        for drive_path in possible_drive_paths:
            if os.path.exists(drive_path):
                word_csv_path = os.path.join(drive_path, "word.csv")
                dataset_csv_path = os.path.join(drive_path, "Dataset_v1.csv")
                
                if os.path.exists(word_csv_path) and os.path.exists(dataset_csv_path):
                    print(f"Found data files in: {drive_path}")
                    return drive_path, drive_path, word_csv_path, dataset_csv_path
                elif os.path.exists(word_csv_path) or os.path.exists(dataset_csv_path):
                    print(f"Partial data found in: {drive_path}")
                    print(f"word.csv exists: {os.path.exists(word_csv_path)}")
                    print(f"Dataset_v1.csv exists: {os.path.exists(dataset_csv_path)}")
    
        if os.path.exists('/content'):
            content_word = "/content/word.csv"
            content_dataset = "/content/Dataset_v1.csv"
            
            if os.path.exists(content_word) and os.path.exists(content_dataset):
                print("Found data files in /content/")
                return "/content", "/content/BrajHindiModelV3", content_word, content_dataset
            else:
                print("Data files not found in /content/ either")
                print(f"Checking: {content_word} - {os.path.exists(content_word)}")
                print(f"Checking: {content_dataset} - {os.path.exists(content_dataset)}")
                return "/content", "/content/BrajHindiModelV3", content_word, content_dataset

        print("🖥️  Local environment detected")
        return ".", "./BrajHindiModelV3", "./word.csv", "./Dataset_v1.csv"
    
    base_dir_str, output_dir_str, word_csv_str, sentence_csv_str = detect_environment()
    
    BASE_DIR = Path(base_dir_str)
    OUTPUT_DIR = Path(output_dir_str)
    WORD_CSV = Path(word_csv_str)
    SENTENCE_CSV = Path(sentence_csv_str)
    
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 2048
    MAX_SEQ_LEN = 128
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    WARMUP_STEPS = 4000
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP = 1.0
    
    VOCAB_SIZE = 4000  
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    USE_AMP = torch.cuda.is_available()

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.word_pairs = []
        self.sentence_pairs = []
        self.word_mapping = {}
        self.sentence_mapping = {}
        
    def load_data(self):
        print("Loading datasets")
        
        with open(self.config.WORD_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    hindi_word = row[0].strip()
                    braj_word = row[1].strip()
                    self.word_pairs.append((hindi_word, braj_word))
                    self.word_mapping[braj_word] = hindi_word
        
        with open(self.config.SENTENCE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    hindi_sent = row[0].strip()
                    braj_sent = row[1].strip()
                    self.sentence_pairs.append((hindi_sent, braj_sent))
                    self.sentence_mapping[braj_sent] = hindi_sent
        
        print(f"Loaded {len(self.word_pairs)} word pairs")
        print(f"Loaded {len(self.sentence_pairs)} sentence pairs")
        
        return self.word_pairs, self.sentence_pairs
    
    def prepare_corpus_files(self):
        braj_corpus_path = self.config.OUTPUT_DIR / "braj_corpus.txt"
        hindi_corpus_path = self.config.OUTPUT_DIR / "hindi_corpus.txt"
        
        with open(braj_corpus_path, 'w', encoding='utf-8') as f:
            for _, braj in self.word_pairs:
                f.write(braj + '\n')
            for _, braj in self.sentence_pairs:
                f.write(braj + '\n')
        
        with open(hindi_corpus_path, 'w', encoding='utf-8') as f:
            for hindi, _ in self.word_pairs:
                f.write(hindi + '\n')
            for hindi, _ in self.sentence_pairs:
                f.write(hindi + '\n')
        
        return str(braj_corpus_path), str(hindi_corpus_path)

class TokenizerManager:
    def __init__(self, config: Config):
        self.config = config
        self.braj_tokenizer = None
        self.hindi_tokenizer = None
        
    def train_tokenizers(self, braj_corpus_path: str, hindi_corpus_path: str):
        print("Training SentencePiece tokenizers")
        
        def get_vocab_size(corpus_path: str) -> int:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            unique_chars = set()
            unique_words = set()
            for line in lines:
                text = line.strip()
                unique_chars.update(text)
                unique_words.update(text.split())
            char_based_vocab = len(unique_chars) * 10  
            word_based_vocab = min(len(unique_words) + 100, self.config.VOCAB_SIZE)
            adaptive_vocab = min(char_based_vocab, word_based_vocab, self.config.VOCAB_SIZE)
            
            print(f"Corpus stats: {len(unique_chars)} unique chars, {len(unique_words)} unique words")
            print(f"Adaptive vocab size: {adaptive_vocab}")
            return max(1000, adaptive_vocab)  
        
        braj_vocab_size = get_vocab_size(braj_corpus_path)
        hindi_vocab_size = get_vocab_size(hindi_corpus_path)
        
        final_vocab_size = min(braj_vocab_size, hindi_vocab_size, 4000)  
        print(f"Final vocabulary size for both tokenizers: {final_vocab_size}")
        braj_model_path = self.config.OUTPUT_DIR / "braj_tokenizer"
        try:
            spm.SentencePieceTrainer.train(
                input=braj_corpus_path,
                model_prefix=str(braj_model_path),
                vocab_size=final_vocab_size,
                character_coverage=1.0,
                model_type='unigram',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
        except RuntimeError as e:
            if "Vocabulary size too high" in str(e):
                import re
                match = re.search(r'<= (\d+)', str(e))
                if match:
                    max_vocab = int(match.group(1))
                    print(f"Adjusting vocab size to maximum allowed: {max_vocab}")
                    final_vocab_size = max_vocab
                    spm.SentencePieceTrainer.train(
                        input=braj_corpus_path,
                        model_prefix=str(braj_model_path),
                        vocab_size=final_vocab_size,
                        character_coverage=1.0,
                        model_type='unigram',
                        pad_id=0,
                        unk_id=1,
                        bos_id=2,
                        eos_id=3
                    )
                else:
                    raise e
            else:
                raise e
            
        hindi_model_path = self.config.OUTPUT_DIR / "hindi_tokenizer"
        spm.SentencePieceTrainer.train(
            input=hindi_corpus_path,
            model_prefix=str(hindi_model_path),
            vocab_size=final_vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        
        self.braj_tokenizer = spm.SentencePieceProcessor()
        self.braj_tokenizer.load(str(braj_model_path) + '.model')
        
        self.hindi_tokenizer = spm.SentencePieceProcessor()
        self.hindi_tokenizer.load(str(hindi_model_path) + '.model')
        
        print(f"✅ Braj vocabulary size: {self.braj_tokenizer.get_piece_size()}")
        print(f"✅ Hindi vocabulary size: {self.hindi_tokenizer.get_piece_size()}")
        
    def encode(self, text: str, tokenizer: spm.SentencePieceProcessor, max_len: int = None) -> List[int]:
        tokens = tokenizer.encode_as_ids(text)
        if max_len:
            tokens = tokens[:max_len-1] + [tokenizer.eos_id()]
            tokens += [tokenizer.pad_id()] * (max_len - len(tokens))
        return tokens
    
    def decode(self, tokens: List[int], tokenizer: spm.SentencePieceProcessor) -> str:
        return tokenizer.decode_ids(tokens)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def get_mask_value(self, dtype):
        if dtype == torch.float16:
            return -1e4  
        else:
            return -1e9  
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            mask_value = self.get_mask_value(attention_scores.dtype)
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.output_projection(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class BrajHindiTransformer(nn.Module):
    def __init__(self, config: Config, braj_vocab_size: int, hindi_vocab_size: int):
        super().__init__()
        self.config = config
        
        self.encoder = TransformerEncoder(braj_vocab_size, config.D_MODEL, config.N_HEADS, config.N_LAYERS, config.D_FF, config.MAX_SEQ_LEN, config.DROPOUT)   
        self.decoder = TransformerDecoder(hindi_vocab_size, config.D_MODEL, config.N_HEADS, config.N_LAYERS, config.D_FF, config.MAX_SEQ_LEN, config.DROPOUT)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output
    
    def create_masks(self, src, tgt, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_subsequent_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        tgt_mask = tgt_mask & tgt_subsequent_mask
        
        return src_mask, tgt_mask

class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer_manager: TokenizerManager, max_len: int = 128):
        self.pairs = pairs
        self.tokenizer_manager = tokenizer_manager
        self.max_len = max_len
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        hindi_text, braj_text = self.pairs[idx]
        braj_tokens = self.tokenizer_manager.encode(braj_text, self.tokenizer_manager.braj_tokenizer, self.max_len)
        hindi_tokens = self.tokenizer_manager.encode(hindi_text, self.tokenizer_manager.hindi_tokenizer, self.max_len)
        hindi_input = [self.tokenizer_manager.hindi_tokenizer.bos_id()] + hindi_tokens[:-1]
        hindi_target = hindi_tokens
        
        return {
            'src': torch.tensor(braj_tokens, dtype=torch.long),
            'tgt_input': torch.tensor(hindi_input, dtype=torch.long),
            'tgt_output': torch.tensor(hindi_target, dtype=torch.long)
        }

class SemanticSimilarityMatcher:
    def __init__(self, sentence_pairs: List[Tuple[str, str]]):
        self.sentence_pairs = sentence_pairs
        self.vectorizer = TfidfVectorizer(max_features=1000)
        braj_sentences = [pair[1] for pair in sentence_pairs]
        self.sentence_embeddings = self.vectorizer.fit_transform(braj_sentences)
        
    def find_similar_sentence(self, query_text: str, top_k: int = 1) -> List[str]:
        query_embedding = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_embedding, self.sentence_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.sentence_pairs[idx][0] for idx in top_indices]  

class Trainer:
    def __init__(self, model: BrajHindiTransformer, config: Config, tokenizer_manager: TokenizerManager):
        self.model = model
        self.config = config
        self.tokenizer_manager = tokenizer_manager
        self.device = config.DEVICE
        self.model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_manager.hindi_tokenizer.pad_id(),label_smoothing=config.LABEL_SMOOTHING)
        self.scaler = GradScaler() if config.USE_AMP else None
        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_bleu = 0.0
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.USE_AMP:
                with autocast():
                    src_mask, tgt_mask = self.model.create_masks(src, tgt_input, self.tokenizer_manager.hindi_tokenizer.pad_id())
                    output = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                src_mask, tgt_mask = self.model.create_masks(src, tgt_input, self.tokenizer_manager.hindi_tokenizer.pad_id())
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        bleu_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                src_mask, tgt_mask = self.model.create_masks(src, tgt_input, self.tokenizer_manager.hindi_tokenizer.pad_id())
                if self.config.USE_AMP:
                    with autocast():
                        output = self.model(src, tgt_input, src_mask, tgt_mask)
                        loss = self.criterion(
                            output.reshape(-1, output.size(-1)),
                            tgt_output.reshape(-1)
                        )
                else:
                    output = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1)
                    )
                
                total_loss += loss.item()
                
                if len(bleu_scores) < 100:  
                    predicted = torch.argmax(output, dim=-1)
                    for i in range(min(5, src.size(0))):  
                        pred_tokens = predicted[i].cpu().tolist()
                        target_tokens = tgt_output[i].cpu().tolist()
                        
                        pred_tokens = [t for t in pred_tokens if t not in [
                            self.tokenizer_manager.hindi_tokenizer.pad_id(),
                            self.tokenizer_manager.hindi_tokenizer.bos_id(),
                            self.tokenizer_manager.hindi_tokenizer.eos_id()
                        ]]
                        target_tokens = [t for t in target_tokens if t not in [
                            self.tokenizer_manager.hindi_tokenizer.pad_id(),
                            self.tokenizer_manager.hindi_tokenizer.bos_id(),
                            self.tokenizer_manager.hindi_tokenizer.eos_id()
                        ]]
                        
                        if pred_tokens and target_tokens:
                            bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
                            bleu_scores.append(bleu)
        
        avg_loss = total_loss / len(val_loader)
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        return avg_loss, avg_bleu
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=self.config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=self.config.NUM_EPOCHS
        )
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            val_loss, val_bleu = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(val_bleu)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val BLEU: {val_bleu:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                self.save_model('best_model.pth')
                print(f"  New best BLEU score: {val_bleu:.4f}")
            if (epoch + 1) % 20 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                self.plot_training_progress()
            
            print("-" * 50)
        self.plot_training_progress()
        self.save_training_history()
    
    def save_model(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_bleu_scores': self.val_bleu_scores,
            'best_bleu': self.best_bleu
        }
        
        torch.save(checkpoint, self.config.OUTPUT_DIR / filename)
        print(f"Model saved: {filename}")
    
    def plot_training_progress(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs, self.val_bleu_scores, 'g-', label='Validation BLEU')
        ax2.set_title('Validation BLEU Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_history(self):
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_bleu': self.val_bleu_scores
        })
        
        history_df.to_csv(self.config.OUTPUT_DIR / 'training_history.csv', index=False)

class InferenceEngine:
    def __init__(self, model: BrajHindiTransformer, tokenizer_manager: TokenizerManager, config: Config, semantic_matcher: SemanticSimilarityMatcher = None):
        self.model = model
        self.tokenizer_manager = tokenizer_manager
        self.config = config
        self.semantic_matcher = semantic_matcher
        self.device = config.DEVICE
        self.model.eval()
    
    def translate(self, braj_text: str, max_length: int = 128) -> str:
        with torch.no_grad():
            src_tokens = self.tokenizer_manager.encode(braj_text, self.tokenizer_manager.braj_tokenizer, max_length)
            src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
            tgt_tokens = [self.tokenizer_manager.hindi_tokenizer.bos_id()]
            
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)
                src_mask, tgt_mask = self.model.create_masks(src_tensor, tgt_tensor, self.tokenizer_manager.hindi_tokenizer.pad_id())
                output = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask)
                next_token = torch.argmax(output[0, -1, :]).item()
                
                if next_token == self.tokenizer_manager.hindi_tokenizer.eos_id():
                    break
                tgt_tokens.append(next_token)
                
            hindi_text = self.tokenizer_manager.decode(
                tgt_tokens[1:], self.tokenizer_manager.hindi_tokenizer
            )
            
            return hindi_text.strip()
    
    def translate_with_fallback(self, braj_text: str) -> str:
        translation = self.translate(braj_text)
        if len(translation.split()) < 2 and self.semantic_matcher:
            similar_translations = self.semantic_matcher.find_similar_sentence(braj_text, top_k=3)
            if similar_translations:
                return f"Direct: {translation} | Similar: {similar_translations[0]}"
        
        return translation

def main():
    print("Starting Braj-Hindi Neural Machine Translation System")
    print("=" * 60)
    
    setup_colab_environment()
    config = Config()
    
    if not config.WORD_CSV.exists() or not config.SENTENCE_CSV.exists():
        print(f"\nData files not found!")
        print(f"Expected file locations:")
        print(f"   - word.csv: {config.WORD_CSV}")
        print(f"   - Dataset_v1.csv: {config.SENTENCE_CSV}")
        print(f"   - Current status:")
        print(f"     word.csv exists: {config.WORD_CSV.exists()}")
        print(f"     Dataset_v1.csv exists: {config.SENTENCE_CSV.exists()}")
        
        if os.path.exists('/content'):
            print(f"\nIn Google Colab:")
            print(f"   Option 1 - Upload to Google Drive:")
            print(f"   1. Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')")
            print(f"   2. Upload files to: /content/drive/MyDrive/Colab Notebooks/BrajHindiModelV3/")
            print(f"   3. Or upload to: /content/drive/MyDrive/Model13/BrajHindiModelV3/")
            print(f"   ")
            print(f"   Option 2 - Upload directly to /content/:")
            print(f"   1. Use Colab's file upload panel")
            print(f"   2. Upload word.csv and Dataset_v1.csv to /content/")
            print(f"   ")
            print(f"   Option 3 - Copy from Drive manually:")
            print(f"   !cp '/content/drive/MyDrive/path/to/your/word.csv' /content/")
            print(f"   !cp '/content/drive/MyDrive/path/to/your/Dataset_v1.csv' /content/")
            
            if os.path.exists('/content/drive/MyDrive'):
                print(f"\nSearching for files in your Google Drive...")
                import glob
                word_matches = glob.glob('/content/drive/MyDrive/**/word.csv', recursive=True)
                if word_matches:
                    print(f"   Found word.csv at: {word_matches[0]}")
                    print(f"   Copy command: !cp '{word_matches[0]}' /content/")
                dataset_matches = glob.glob('/content/drive/MyDrive/**/Dataset_v1.csv', recursive=True)
                if dataset_matches:
                    print(f"   Found Dataset_v1.csv at: {dataset_matches[0]}")
                    print(f"   Copy command: !cp '{dataset_matches[0]}' /content/")
                
                if not word_matches and not dataset_matches:
                    print(f"   No matching files found in Google Drive")
        else:
            print(f"\nIn Local Environment:")
            print(f"   Ensure files are in the same directory as this script:")
            print(f"   - word.csv")
            print(f"   - Dataset_v1.csv")
        
        return
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {config.OUTPUT_DIR}")
    preprocessor = DataPreprocessor(config)
    tokenizer_manager = TokenizerManager(config)
    
    print("\nLoading and preprocessing data...")
    word_pairs, sentence_pairs = preprocessor.load_data()
    braj_corpus_path, hindi_corpus_path = preprocessor.prepare_corpus_files()
    print("\nTraining SentencePiece tokenizers...")
    tokenizer_manager.train_tokenizers(braj_corpus_path, hindi_corpus_path)
    print("\nPreparing datasets...")
    all_pairs = word_pairs + sentence_pairs
    random.shuffle(all_pairs)
    train_size = int(0.8 * len(all_pairs))
    val_size = int(0.1 * len(all_pairs))
    
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:train_size + val_size]
    test_pairs = all_pairs[train_size + val_size:]
    
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    
    train_dataset = TranslationDataset(train_pairs, tokenizer_manager, config.MAX_SEQ_LEN)
    val_dataset = TranslationDataset(val_pairs, tokenizer_manager, config.MAX_SEQ_LEN)
    test_dataset = TranslationDataset(test_pairs, tokenizer_manager, config.MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print("\nInitializing Transformer model...")
    braj_vocab_size = tokenizer_manager.braj_tokenizer.get_piece_size()
    hindi_vocab_size = tokenizer_manager.hindi_tokenizer.get_piece_size()
    
    model = BrajHindiTransformer(config, braj_vocab_size, hindi_vocab_size)
    trainer = Trainer(model, config, tokenizer_manager)

    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    print("\nSaving mappings and final results...")
    
    with open(config.OUTPUT_DIR / 'word_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(preprocessor.word_mapping, f, ensure_ascii=False, indent=2)
    
    with open(config.OUTPUT_DIR / 'sentence_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(preprocessor.sentence_mapping, f, ensure_ascii=False, indent=2)
    semantic_matcher = SemanticSimilarityMatcher(sentence_pairs)
    print("\nTesting inference")
    inference_engine = InferenceEngine(model, tokenizer_manager, config, semantic_matcher)
    test_examples = [
        "हौं खाना खात हौं",
        "सूरज उगत है",
        "बच्चे खेलत हैं"
    ]
    test_results = {}
    for braj_text in test_examples:
        translation = inference_engine.translate_with_fallback(braj_text)
        test_results[braj_text] = translation
        print(f"Braj: {braj_text}")
        print(f"Hindi: {translation}")
        print("-" * 30)
    with open(config.OUTPUT_DIR / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print("\nTraining completed successfully!")
    print(f"All outputs saved in: {config.OUTPUT_DIR}")
    print(f"Best validation BLEU score: {trainer.best_bleu:.4f}")
    print("\nFinal evaluation on test set...")
    test_loss, test_bleu = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU: {test_bleu:.4f}")
    final_results = {
        'best_val_bleu': trainer.best_bleu,
        'test_loss': test_loss,
        'test_bleu': test_bleu,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_epochs': config.NUM_EPOCHS,
        'device_used': str(config.DEVICE)
    }
    with open(config.OUTPUT_DIR / 'final_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print("\nBraj-Hindi Translation System is ready for use!")

if __name__ == "__main__":
    main()
