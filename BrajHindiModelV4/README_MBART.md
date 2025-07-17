# mBART Fine-tuning for Braj-Hindi Translation

A high-performance translation system using Facebook's mBART-large model fine-tuned for Hindi to Braj translation.

## 🚀 Quick Start

```bash
python run_mbart.py
```

Choose from the menu options to:
1. Run full pipeline (recommended for first time)
2. Fine-tune only
3. Test existing model
4. Interactive translation
5. Install dependencies only

## 📁 Project Structure

```
BrajHindiModelV4/
├── Dataset_v1.csv              # Training data (7000+ pairs)
├── mbart_fine_tune.py          # Main fine-tuning script
├── mbart_inference.py          # Inference and testing
├── setup_mbart.py              # Full pipeline setup
├── run_mbart.py                # Quick start script
└── fine_tuned_mbart_braj_hindi/ # Output model directory
```

## 🔧 System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 5GB+ free space
- **GPU**: CUDA-compatible (recommended for training)
- **Internet**: Required for downloading mBART model

## 📦 Dependencies

```bash
pip install torch transformers datasets pandas numpy scikit-learn tqdm accelerate evaluate
```

## 🎯 Usage Examples

### Fine-tuning
```bash
python mbart_fine_tune.py
```

### Interactive Translation
```bash
python mbart_inference.py --interactive
```

### Single Translation
```bash
python mbart_inference.py --translate "मैं घर जा रहा हूं"
```

### Batch Translation
```bash
python mbart_inference.py --batch input.txt output.json
```

### Performance Testing
```bash
python mbart_inference.py --test
```

## 📊 Model Specifications

- **Base Model**: facebook/mbart-large-50-many-to-many-mmt
- **Parameters**: ~600M
- **Training Data**: 7000+ Hindi-Braj sentence pairs
- **Training Time**: 2-4 hours (GPU) / 8-12 hours (CPU)
- **Inference Speed**: ~0.1-0.5s per translation
- **Memory Usage**: 2-4GB GPU / 4-8GB RAM

## 🎛️ Configuration Options

### Fine-tuning Parameters
- **Epochs**: 5 (adjustable in `mbart_fine_tune.py`)
- **Batch Size**: 4 (adjust based on GPU memory)
- **Learning Rate**: 3e-5
- **Max Length**: 128 tokens
- **Gradient Accumulation**: 4 steps

### Inference Parameters
- **Beam Search**: 3 beams (adjustable)
- **Temperature**: 0.8 (creativity level)
- **Max Length**: 128 tokens

## 📈 Performance Metrics

Expected performance after fine-tuning:
- **Translation Quality**: Significantly improved over word-mapping
- **Fluency**: Natural Braj sentences
- **Speed**: Real-time translation capability
- **Coverage**: Handles complex sentences and context

## 🔍 Evaluation

The system includes comprehensive evaluation:
- **BLEU Score**: Automatic evaluation metric
- **Translation Time**: Performance benchmarking
- **Sample Translations**: Manual quality check
- **Error Analysis**: Detailed reporting

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in `mbart_fine_tune.py`
   - Use gradient checkpointing
   - Close other applications

2. **Slow Training**
   - Enable CUDA if available
   - Reduce max_length parameter
   - Use mixed precision training

3. **Poor Translation Quality**
   - Increase training epochs
   - Check data quality
   - Adjust learning rate

4. **Model Not Found**
   - Run fine-tuning first
   - Check output directory path
   - Verify model saving completed

### GPU Setup

For CUDA setup:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

## 📋 Dataset Format

The training data should be in CSV format:
```csv
hindi_sentence,braj_translation
मैं जाता हूं,हौं जात हौं
यह अच्छा है,यह भलो है
```

## 🎨 Customization

### Adding More Data
1. Add new Hindi-Braj pairs to `Dataset_v1.csv`
2. Re-run fine-tuning with updated dataset
3. Evaluate improved model performance

### Adjusting Translation Style
- Modify `temperature` parameter for creativity
- Adjust `num_beams` for translation quality
- Fine-tune with domain-specific data

## 📞 Support

For issues or questions:
1. Check this README for common solutions
2. Review error messages in console output
3. Verify system requirements and dependencies

## 🏆 Results

After successful fine-tuning, you'll have:
- A specialized Braj-Hindi translation model
- Real-time translation capability
- Comprehensive evaluation metrics
- Interactive translation interface

## 🚀 Next Steps

1. **Run the quick start**: `python run_mbart.py`
2. **Fine-tune the model**: Choose option 1 or 2
3. **Test performance**: Use option 3
4. **Start translating**: Use option 4 for interactive mode

The fine-tuned model will be saved in `./fine_tuned_mbart_braj_hindi/` and ready for production use!
