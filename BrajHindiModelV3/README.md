# 🚀 Braj-Hindi Neural Machine Translation System

A powerful, GPT-inspired transformer-based neural machine translation system for translating between Braj and Hindi languages. Built with PyTorch and featuring state-of-the-art architecture with multi-head self-attention, positional encoding, and advanced training techniques.

## 🌟 Features

- **GPT-style Transformer Architecture**: 6-layer encoder-decoder with multi-head attention
- **SentencePiece Tokenization**: Custom tokenizers for both Braj and Hindi
- **Mixed Precision Training**: FP16 support for faster GPU training
- **Semantic Similarity Fallback**: Handles unseen words using cosine similarity
- **Comprehensive Evaluation**: BLEU, ROUGE, METEOR metrics
- **Interactive Inference**: Easy-to-use translation interface
- **Checkpoint Management**: Automatic saving every 20 epochs
- **Training Visualization**: Loss and BLEU score plots

## 📁 Project Structure

```
BrajHindiModelV3/
├── braj_hindi_transformer.py    # Main training script
├── inference.py                 # Inference interface
├── evaluation.py               # Comprehensive evaluation
├── config.json                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── best_model.pth             # Best trained model
├── braj_tokenizer.model       # Braj tokenizer
├── hindi_tokenizer.model      # Hindi tokenizer
├── word_mapping.json          # Word-level mappings
├── sentence_mapping.json      # Sentence-level mappings
├── training_history.csv       # Training metrics
├── training_analysis.png      # Training plots
└── evaluation_results/        # Evaluation outputs
    ├── detailed_translations.csv
    ├── evaluation_metrics.json
    ├── evaluation_summary.txt
    └── evaluation_distributions.png
```

## 🛠️ Installation

1. **Clone and navigate to the project:**

```bash
cd BrajHindiModelV3
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download NLTK data (required for evaluation):**

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🔧 Configuration

The model can be configured through `config.json`:

```json
{
  "model_config": {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 2048,
    "max_seq_len": 128,
    "dropout": 0.1,
    "vocab_size": 8000
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 200,
    "warmup_steps": 4000,
    "label_smoothing": 0.1
  }
}
```

## 🏋️ Training

### Quick Start Training

```bash
python braj_hindi_transformer.py
```

### Training Features

- **200 epochs** with automatic early stopping
- **Mixed precision training** for faster GPU utilization
- **OneCycleLR scheduler** with warmup
- **Label smoothing** for better generalization
- **Gradient clipping** for stable training
- **Automatic checkpointing** every 20 epochs

### Training Output

The training script will create:

- Model checkpoints (`checkpoint_epoch_20.pth`, etc.)
- Best model (`best_model.pth`)
- Training metrics (`training_history.csv`)
- Visualization plots (`training_analysis.png`)

## 🔍 Inference

### Interactive Mode

```bash
python inference.py --interactive
```

### Single Translation

```bash
python inference.py --text "हौं खाना खात हौं"
```

### Batch Translation

```bash
python inference.py --file input_texts.txt
```

### Programmatic Usage

```python
from inference import InferenceInterface

# Initialize translator
translator = InferenceInterface("BrajHindiModelV3")

# Translate text
result = translator.translate("हौं खाना खात हौं")
print(result['recommended'])  # Neural translation

# Interactive mode
translator.interactive_mode()
```

## 📊 Evaluation

### Comprehensive Evaluation

```bash
python evaluation.py --model_dir BrajHindiModelV3
```

### Limited Sample Evaluation

```bash
python evaluation.py --max_samples 1000
```

### Evaluation Metrics

- **BLEU Score**: Translation quality metric
- **ROUGE-1/2/L**: Text overlap metrics
- **METEOR**: Semantic similarity metric
- **Exact Match**: Perfect translation accuracy
- **Word Overlap**: Jaccard similarity
- **Length Statistics**: Translation length analysis

## 🏗️ Model Architecture

### Transformer Components

1. **Encoder Stack** (6 layers):

   - Multi-head self-attention (8 heads)
   - Position-wise feed-forward networks
   - Residual connections and layer normalization
   - Positional encoding

2. **Decoder Stack** (6 layers):

   - Masked multi-head self-attention
   - Encoder-decoder attention
   - Position-wise feed-forward networks
   - Output projection to vocabulary

3. **Training Features**:
   - Teacher forcing during training
   - Label smoothing (0.1)
   - Dropout (0.1) for regularization
   - Mixed precision training

### Model Parameters

- **Total Parameters**: ~25M (estimated)
- **Hidden Dimension**: 512
- **Feed-forward Dimension**: 2048
- **Attention Heads**: 8
- **Layers**: 6 (encoder) + 6 (decoder)

## 📈 Performance

### Expected Results

Based on the architecture and dataset:

- **BLEU Score**: 0.35-0.45 (good to excellent)
- **Training Time**: ~8-12 hours on GPU
- **Inference Speed**: ~50-100 translations/second
- **Memory Usage**: ~4-6 GB GPU memory

### Optimization Features

- **Mixed Precision**: 40-50% speedup on modern GPUs
- **Efficient Attention**: Scaled dot-product attention
- **Batch Processing**: Optimized for parallel inference
- **Caching**: Tokenizer and model caching for speed

## 🔧 Advanced Usage

### Custom Training

```python
from braj_hindi_transformer import Config, Trainer

# Modify configuration
config = Config()
config.NUM_EPOCHS = 100
config.BATCH_SIZE = 64

# Train with custom config
# ... (see main training script for full example)
```

### Model Fine-tuning

```python
# Load pre-trained model
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training with lower learning rate
config.LEARNING_RATE = 1e-5
```

### Custom Evaluation

```python
from evaluation import TranslationEvaluator

evaluator = TranslationEvaluator("BrajHindiModelV3")
results, aggregates = evaluator.run_full_evaluation()

# Access specific metrics
bleu_score = aggregates['bleu']['mean']
rouge_score = aggregates['rouge1']['mean']
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

   - Reduce batch size in config
   - Enable mixed precision training
   - Use gradient accumulation

2. **Slow Training**:

   - Enable mixed precision (`USE_AMP = True`)
   - Increase batch size if memory allows
   - Use multiple GPUs with DataParallel

3. **Poor Translation Quality**:
   - Increase training epochs
   - Add more training data
   - Adjust learning rate scheduler
   - Fine-tune on domain-specific data

### Performance Optimization

```bash
# Monitor GPU usage
nvidia-smi

# Profile training
python -m torch.profiler braj_hindi_transformer.py
```

## 📚 Dataset Format

### Word Pairs (word.csv)

```csv
hindi,braj
मैं,हौं
तुम,तुम
वह,वह
```

### Sentence Pairs (Dataset_v1.csv)

```csv
hindi_sentence,braj_translation
मैं सुबह छह बजे उठता हूँ,हौं भोर छह बजे जागत हौं
सूरज की पहली किरण से मेरी नींद खुलती है,सूरज की पहली किरन तें मेरी निद्रा खुलत है
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **SentencePiece**: For subword tokenization
- **Transformer Architecture**: Attention Is All You Need (Vaswani et al.)
- **NLTK**: For evaluation metrics

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the evaluation metrics for model performance

---

**Built with ❤️ for preserving and translating Braj language using state-of-the-art AI**
