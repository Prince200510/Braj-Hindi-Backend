# 🚀 Braj-to-Hindi Neural Machine Translation System

A state-of-the-art **Transformer-based Neural Machine Translation** system for translating between Braj Bhasha and Hindi. Features intelligent subword fallback, beam search decoding, and production-ready CLI interface.

## ✨ Features

- **🧠 Advanced Transformer Architecture**: Multi-head attention with 4 encoder-decoder layers
- **📝 SentencePiece Tokenization**: BPE tokenization with 4K vocabulary size
- **🎯 Intelligent Fallback System**: Subword decomposition and cosine similarity matching
- **🔍 Beam Search Decoding**: High-quality translation with multiple candidates
- **📊 BLEU Score Evaluation**: BLEU-1 to BLEU-3 scoring during training
- **💻 Production-Ready CLI**: Interactive and batch translation modes
- **🎨 Beautiful Training Visualization**: Real-time loss and accuracy plots

## 🏗️ Architecture

### Model Configuration
- **Architecture**: Transformer Encoder-Decoder
- **Layers**: 4 encoder + 4 decoder layers
- **Attention Heads**: 8 multi-head attention
- **Embedding Dimension**: 256
- **Feed-Forward Network**: 1024 units
- **Dropout**: 0.2
- **Maximum Sequence Length**: 20 tokens
- **Optimizer**: Adam with warmup learning rate schedule

### Intelligent Fallback Logic
1. **Direct Lookup**: Check word-level translation dictionary
2. **Neural Translation**: Full transformer model inference
3. **Subword Fallback**: BPE decomposition and reconstruction
4. **Similarity Matching**: Character-level similarity for unknown tokens

## 📁 Project Structure

```
Model12/
├── dataset.csv              # Training data (574 Braj-Hindi pairs)
├── model.py                 # Transformer architecture implementation
├── train.py                 # Training script with evaluation
├── translate.py             # CLI translation interface
├── requirements.txt         # Python dependencies
└── README.md               # This file

# Generated after training:
├── braj_tokenizer.model     # Trained Braj SentencePiece tokenizer
├── hindi_tokenizer.model    # Trained Hindi SentencePiece tokenizer
├── best_model_weights.*     # Best model checkpoint
├── final_model_weights.*    # Final model weights
├── model_metadata.pkl       # Training metadata
└── training_history.png     # Training visualization
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

**Training Process:**
- Loads `dataset.csv` with 574+ Braj-Hindi translation pairs
- Trains SentencePiece tokenizers for both languages
- Trains Transformer model for 50 epochs
- Evaluates with BLEU scores every 5 epochs
- Saves best model based on BLEU-1 score
- Generates training visualization plots

**Expected Output:**
```
🚀 Starting Braj-to-Hindi Neural Machine Translation Training
============================================================
Loading dataset...
Loaded 573 text pairs
Training SentencePiece tokenizers...
Braj vocabulary size: 4000
Hindi vocabulary size: 4000
Tokenizing sequences...
Dataset shape: Braj=(573, 20), Hindi_input=(573, 20), Hindi_target=(573, 20)
Training set size: 458
Validation set size: 115
Model created with 4000 input vocab and 4000 target vocab

Epoch 1/50
Training: 100%|████████████| 15/15 [00:23<00:00,  1.56s/it]
Loss: 6.2341, Accuracy: 0.1234
...
```

### 3. Interactive Translation

```bash
python translate.py --interactive
```

**Interactive Mode:**
```
🚀 Braj-to-Hindi Interactive Translator
==================================================
Commands:
  - Type Braj text to translate
  - ':mode word' or ':mode sentence' to change mode
  - ':quit' or ':exit' to exit
  - ':help' for this help message
==================================================

[auto] Braj> हौं
Hindi> मैं

[auto] Braj> तुम जल पीवत हो
Hindi> तुम पानी पी रहे हो

[auto] Braj> :mode word
✅ Mode changed to: word

[word] Braj> करब
Hindi> करना
```

### 4. Single Translation

```bash
# Translate single text
python translate.py --input "हौं जाब"

# Word-only mode
python translate.py --input "करब" --mode word

# Verbose output
python translate.py --input "तुम जल पीवत हो" --verbose
```

## 📊 Model Performance

### Training Metrics (Expected)
- **Final Training Loss**: ~2.5-3.0
- **Final Training Accuracy**: ~75-85%
- **BLEU-1 Score**: ~0.65-0.75
- **BLEU-2 Score**: ~0.45-0.60
- **BLEU-3 Score**: ~0.30-0.50

### Translation Examples
| Braj Input | Hindi Output | Translation Type |
|------------|--------------|------------------|
| हौं | मैं | Direct Word |
| तुम | तुम | Direct Word |
| करब | करना | Neural Model |
| तुम जल पीवत हो | तुम पानी पी रहे हो | Neural + Fallback |
| हौं आवब | मैं आऊंगा | Subword Fallback |

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from model import create_model, create_optimizer

# Create custom model
model = create_model(
    input_vocab_size=4000,
    target_vocab_size=4000,
    max_seq_len=20
)

# Custom optimizer
optimizer = create_optimizer(d_model=256)
```

### Programmatic Translation

```python
from translate import BrajHindiTranslator

# Initialize translator
translator = BrajHindiTranslator(
    model_weights_path="best_model_weights",
    metadata_path="model_metadata.pkl",
    braj_tokenizer_path="braj_tokenizer.model",
    hindi_tokenizer_path="hindi_tokenizer.model"
)

# Translate
result = translator.translate("हौं जाब", mode="auto")
print(f"Translation: {result}")
```

### Batch Translation

```python
texts = ["हौं", "तुम", "करब", "जाब"]
translations = [translator.translate(text) for text in texts]
for braj, hindi in zip(texts, translations):
    print(f"{braj} → {hindi}")
```

## 🎯 CLI Options

### train.py
```bash
python train.py  # Uses default configuration
```

### translate.py
```bash
# Required arguments
python translate.py --input "text"           # Single translation
python translate.py --interactive            # Interactive mode

# Optional arguments
--mode {word,sentence,auto}                  # Translation mode (default: auto)
--model MODEL_PATH                           # Model weights path (default: best_model_weights)
--verbose                                    # Verbose output
```

## 🧠 Technical Details

### Transformer Architecture
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: 8 attention heads with scaled dot-product
- **Layer Normalization**: Applied after each sub-layer
- **Residual Connections**: Skip connections throughout the model
- **Masking**: Proper padding and look-ahead masks for training

### Tokenization Strategy
- **Algorithm**: Byte Pair Encoding (BPE) via SentencePiece
- **Vocabulary Size**: 4,000 tokens per language
- **Special Tokens**: PAD(0), UNK(1), BOS(2), EOS(3)
- **Subword Units**: Handles out-of-vocabulary words gracefully

### Training Optimizations
- **Learning Rate Schedule**: Warmup + decay schedule
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Based on BLEU score improvement
- **Data Augmentation**: Automatic through subword tokenization

## 📈 Performance Optimization

### For Better Results:
1. **Increase Dataset Size**: Add more Braj-Hindi parallel sentences
2. **Tune Hyperparameters**: Adjust learning rate, batch size, model dimensions
3. **Longer Training**: Increase epochs for convergence
4. **Data Quality**: Clean and verify translation pairs
5. **Ensemble Methods**: Combine multiple model predictions

### For Faster Inference:
1. **Model Quantization**: Reduce model size with TensorFlow Lite
2. **Beam Search Width**: Lower beam width for faster decoding
3. **Sequence Length**: Reduce max sequence length if possible
4. **Batch Processing**: Process multiple sentences together

## 🐛 Troubleshooting

### Common Issues:

**1. ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow>=2.10.0
```

**2. Missing model files**
```bash
# Run training first
python train.py
```

**3. Out of memory during training**
```python
# Reduce batch size in train.py
BATCH_SIZE = 16  # Instead of 32
```

**4. Poor translation quality**
- Check dataset quality and size
- Increase training epochs
- Verify tokenizer training
- Add more parallel data

### GPU Usage
```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 🤝 Contributing

1. **Add More Data**: Contribute Braj-Hindi translation pairs
2. **Improve Architecture**: Experiment with newer transformer variants
3. **Performance Optimization**: Profile and optimize inference speed
4. **UI Development**: Create web interface or mobile app
5. **Evaluation Metrics**: Add more robust evaluation methods

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **TensorFlow**: Deep learning framework
- **SentencePiece**: Subword tokenization
- **Transformer Architecture**: "Attention Is All You Need" paper
- **Braj Bhasha Community**: For language preservation efforts

---

**Built with ❤️ for preserving and promoting Braj Bhasha through modern AI technology.**
