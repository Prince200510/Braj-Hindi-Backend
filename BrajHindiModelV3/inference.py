import os
import json
import torch
import sentencepiece as spm
from pathlib import Path
import argparse

from braj_hindi_transformer import (Config, BrajHindiTransformer, TokenizerManager, InferenceEngine, SemanticSimilarityMatcher)

class InferenceInterface:
    def __init__(self, model_dir: str = "."):
        self.model_dir = Path(model_dir)
        self.config = Config()
        self.config.OUTPUT_DIR = self.model_dir
        self.load_model()
        self.load_mappings()
        self.inference_engine = InferenceEngine(self.model, self.tokenizer_manager, self.config, self.semantic_matcher)
        
    def load_model(self):
        model_path = self.model_dir / "best_model.pth"
        
        print(f"Looking for model at: {model_path.absolute()}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print("Loading trained model...")
        try:
            checkpoint = torch.load(model_path, map_location=self.config.DEVICE, weights_only=False)
        except AttributeError as e:
            if "Config" in str(e):
                print("Handling Config class import issue...")
                import braj_hindi_transformer as main_module
                import sys
                sys.modules['__main__'].Config = main_module.Config
                checkpoint = torch.load(model_path, map_location=self.config.DEVICE, weights_only=False)
            else:
                raise e
        self.load_tokenizers()
        braj_vocab_size = self.tokenizer_manager.braj_tokenizer.get_piece_size()
        hindi_vocab_size = self.tokenizer_manager.hindi_tokenizer.get_piece_size()
        
        print(f"Braj vocab size: {braj_vocab_size}, Hindi vocab size: {hindi_vocab_size}")
        
        self.model = BrajHindiTransformer(self.config, braj_vocab_size, hindi_vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.DEVICE)
        self.model.eval()
        
        print(f"Model loaded successfully! Best BLEU: {checkpoint.get('best_bleu', 'N/A'):.4f}")
        
    def load_tokenizers(self):
        braj_tokenizer_path = self.model_dir / "braj_tokenizer.model"
        hindi_tokenizer_path = self.model_dir / "hindi_tokenizer.model"
        
        if not braj_tokenizer_path.exists() or not hindi_tokenizer_path.exists():
            raise FileNotFoundError("Tokenizer models not found")
        
        print("Loading tokenizers...")
        self.tokenizer_manager = TokenizerManager(self.config)
        self.tokenizer_manager.braj_tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_manager.braj_tokenizer.load(str(braj_tokenizer_path))
        
        self.tokenizer_manager.hindi_tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_manager.hindi_tokenizer.load(str(hindi_tokenizer_path))
        
        print("Tokenizers loaded successfully!")
        
    def load_mappings(self):
        word_mapping_path = self.model_dir / "word_mapping.json"
        sentence_mapping_path = self.model_dir / "sentence_mapping.json"
        
        self.word_mapping = {}
        self.sentence_pairs = []
        
        if word_mapping_path.exists():
            with open(word_mapping_path, 'r', encoding='utf-8') as f:
                self.word_mapping = json.load(f)
        
        if sentence_mapping_path.exists():
            with open(sentence_mapping_path, 'r', encoding='utf-8') as f:
                sentence_mapping = json.load(f)
                self.sentence_pairs = [(v, k) for k, v in sentence_mapping.items()]
        if self.sentence_pairs:
            self.semantic_matcher = SemanticSimilarityMatcher(self.sentence_pairs)
        else:
            self.semantic_matcher = None
        
        print(f"Loaded {len(self.word_mapping)} word mappings")
        print(f"Loaded {len(self.sentence_pairs)} sentence pairs for semantic matching")
        
    def translate(self, braj_text: str, use_semantic_fallback: bool = True) -> dict:
        direct_translation = self.word_mapping.get(braj_text.strip())
        
        if use_semantic_fallback:
            neural_translation = self.inference_engine.translate_with_fallback(braj_text)
        else:
            neural_translation = self.inference_engine.translate(braj_text)
        
        return {
            'input': braj_text,
            'direct_mapping': direct_translation,
            'neural_translation': neural_translation,
            'recommended': direct_translation if direct_translation else neural_translation
        }
    
    def interactive_mode(self):
        print("\nInteractive Braj-Hindi Translation")
        print("Enter Braj text to translate (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            braj_input = input("\nBraj: ").strip()
            
            if braj_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not braj_input:
                continue
            
            try:
                result = self.translate(braj_input)
                print(f"Hindi: {result['recommended']}")
                
                if result['direct_mapping'] and result['neural_translation']:
                    print(f"Direct: {result['direct_mapping']}")
                    print(f"Neural: {result['neural_translation']}")
                
            except Exception as e:
                print(f"Error during translation: {e}")
    
    def batch_translate(self, texts: list) -> list:
        results = []
        for text in texts:
            try:
                result = self.translate(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'input': text,
                    'error': str(e),
                    'recommended': f"Error: {e}"
                })
        return results

def main():
    parser = argparse.ArgumentParser(description="Braj-Hindi Translation Inference")
    parser.add_argument("--model_dir", type=str, default="BrajHindiModelV3",help="Directory containing the trained model")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--file", type=str, help="File containing texts to translate")
    
    args = parser.parse_args()
    
    try:
        translator = InferenceInterface(args.model_dir)
        
        if args.interactive or (not args.text and not args.file):
            translator.interactive_mode()
            
        elif args.text:
            result = translator.translate(args.text)
            print(f"Braj: {result['input']}")
            print(f"Hindi: {result['recommended']}")
            
        elif args.file:
            if not Path(args.file).exists():
                print(f"File not found: {args.file}")
                return
            
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            results = translator.batch_translate(texts)
            
            output_file = Path(args.file).stem + "_translated.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"Translated {len(results)} texts")
            print(f"Results saved to: {output_file}")
    
            for i, result in enumerate(results[:5]):
                print(f"{i+1}. {result['input']} → {result['recommended']}")
            
            if len(results) > 5:
                print(f"... and {len(results)-5} more")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
