import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Install with: pip install rouge-score")

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

from braj_hindi_transformer import (Config, BrajHindiTransformer, TokenizerManager, TranslationDataset, DataPreprocessor, InferenceEngine)

class TranslationEvaluator:
    def __init__(self, model_dir: str = "BrajHindiModelV3"):
        self.model_dir = Path(model_dir)
        self.config = Config()
        self.config.OUTPUT_DIR = self.model_dir
    
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth_func = SmoothingFunction().method1
        
    def load_model_and_data(self):
        print("Loading model and data")
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        braj_vocab_size = 8000  
        hindi_vocab_size = 8000
        
        self.model = BrajHindiTransformer(self.config, braj_vocab_size, hindi_vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.DEVICE)
        self.model.eval()
        self.tokenizer_manager = TokenizerManager(self.config)
        
        import sentencepiece as spm
        self.tokenizer_manager.braj_tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_manager.braj_tokenizer.load(str(self.model_dir / "braj_tokenizer.model"))
        
        self.tokenizer_manager.hindi_tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_manager.hindi_tokenizer.load(str(self.model_dir / "hindi_tokenizer.model"))
        preprocessor = DataPreprocessor(self.config)
        word_pairs, sentence_pairs = preprocessor.load_data()
        all_pairs = word_pairs + sentence_pairs
        train_size = int(0.8 * len(all_pairs))
        val_size = int(0.1 * len(all_pairs))
        self.test_pairs = all_pairs[train_size + val_size:]
        
        print(f"Loaded {len(self.test_pairs)} test pairs")

        self.inference_engine = InferenceEngine(self.model, self.tokenizer_manager, self.config)
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smooth_func)
        return bleu
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        if not ROUGE_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_meteor_score(self, reference: str, hypothesis: str) -> float:
        try:
            try:
                nltk.data.find('wordnet')
            except LookupError:
                nltk.download('wordnet')
            
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            
            if not ref_tokens or not hyp_tokens:
                return 0.0
            
            return meteor_score([ref_tokens], hyp_tokens)
        except:
            return 0.0
    
    def calculate_exact_match(self, reference: str, hypothesis: str) -> float:
        return 1.0 if reference.strip() == hypothesis.strip() else 0.0
    
    def calculate_word_overlap(self, reference: str, hypothesis: str) -> float:
        ref_words = set(reference.split())
        hyp_words = set(hypothesis.split())
        
        if not ref_words and not hyp_words:
            return 1.0
        if not ref_words or not hyp_words:
            return 0.0
        
        intersection = ref_words.intersection(hyp_words)
        union = ref_words.union(hyp_words)
        
        return len(intersection) / len(union)
    
    def evaluate_model(self, max_samples: int = None) -> Dict:
        print("Starting comprehensive evaluation...")
        test_pairs = self.test_pairs[:max_samples] if max_samples else self.test_pairs
        
        results = {
            'translations': [],
            'bleu_scores': [],
            'rouge_scores': [],
            'meteor_scores': [],
            'exact_matches': [],
            'word_overlaps': [],
            'translation_lengths': [],
            'reference_lengths': []
        }
        for i, (hindi_ref, braj_input) in enumerate(tqdm(test_pairs, desc="Evaluating")):
            try:
                hindi_pred = self.inference_engine.translate(braj_input)
                bleu = self.calculate_bleu_score(hindi_ref, hindi_pred)
                rouge = self.calculate_rouge_scores(hindi_ref, hindi_pred)
                meteor = self.calculate_meteor_score(hindi_ref, hindi_pred)
                exact_match = self.calculate_exact_match(hindi_ref, hindi_pred)
                word_overlap = self.calculate_word_overlap(hindi_ref, hindi_pred)
                
                results['translations'].append({
                    'braj_input': braj_input,
                    'hindi_reference': hindi_ref,
                    'hindi_prediction': hindi_pred,
                    'bleu': bleu,
                    'rouge1': rouge['rouge1'],
                    'rouge2': rouge['rouge2'],
                    'rougeL': rouge['rougeL'],
                    'meteor': meteor,
                    'exact_match': exact_match,
                    'word_overlap': word_overlap
                })
                
                results['bleu_scores'].append(bleu)
                results['rouge_scores'].append(rouge)
                results['meteor_scores'].append(meteor)
                results['exact_matches'].append(exact_match)
                results['word_overlaps'].append(word_overlap)
                results['translation_lengths'].append(len(hindi_pred.split()))
                results['reference_lengths'].append(len(hindi_ref.split()))
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
        
        return results
    
    def calculate_aggregate_metrics(self, results: Dict) -> Dict:
        aggregates = {}
        bleu_scores = results['bleu_scores']
        aggregates['bleu'] = {
            'mean': np.mean(bleu_scores),
            'std': np.std(bleu_scores),
            'median': np.median(bleu_scores),
            'min': np.min(bleu_scores),
            'max': np.max(bleu_scores)
        }
        if ROUGE_AVAILABLE:
            rouge1_scores = [r['rouge1'] for r in results['rouge_scores']]
            rouge2_scores = [r['rouge2'] for r in results['rouge_scores']]
            rougeL_scores = [r['rougeL'] for r in results['rouge_scores']]
            
            aggregates['rouge1'] = {
                'mean': np.mean(rouge1_scores),
                'std': np.std(rouge1_scores)
            }
            aggregates['rouge2'] = {
                'mean': np.mean(rouge2_scores),
                'std': np.std(rouge2_scores)
            }
            aggregates['rougeL'] = {
                'mean': np.mean(rougeL_scores),
                'std': np.std(rougeL_scores)
            }
        meteor_scores = results['meteor_scores']
        aggregates['meteor'] = {
            'mean': np.mean(meteor_scores),
            'std': np.std(meteor_scores)
        }
        aggregates['exact_match_accuracy'] = np.mean(results['exact_matches'])
        aggregates['word_overlap'] = {
            'mean': np.mean(results['word_overlaps']),
            'std': np.std(results['word_overlaps'])
        }
        aggregates['length_stats'] = {
            'avg_prediction_length': np.mean(results['translation_lengths']),
            'avg_reference_length': np.mean(results['reference_lengths']),
            'length_ratio': np.mean(results['translation_lengths']) / np.mean(results['reference_lengths'])
        }
        
        return aggregates
    
    def create_evaluation_plots(self, results: Dict, output_dir: Path):
        print("Creating evaluation plots...")
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].hist(results['bleu_scores'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('BLEU Score Distribution')
        axes[0, 0].set_xlabel('BLEU Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(results['bleu_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(results["bleu_scores"]):.3f}')
        axes[0, 0].legend()
    
        if ROUGE_AVAILABLE:
            rouge1_scores = [r['rouge1'] for r in results['rouge_scores']]
            axes[0, 1].hist(rouge1_scores, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('ROUGE-1 Score Distribution')
            axes[0, 1].set_xlabel('ROUGE-1 Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(rouge1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rouge1_scores):.3f}')
            axes[0, 1].legend()
        
        axes[0, 2].hist(results['meteor_scores'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('METEOR Score Distribution')
        axes[0, 2].set_xlabel('METEOR Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(results['meteor_scores']), color='red', linestyle='--', label=f'Mean: {np.mean(results["meteor_scores"]):.3f}')
        axes[0, 2].legend()
        
        axes[1, 0].hist(results['word_overlaps'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Word Overlap Distribution')
        axes[1, 0].set_xlabel('Word Overlap (Jaccard)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(results['word_overlaps']), color='red', linestyle='--', label=f'Mean: {np.mean(results["word_overlaps"]):.3f}')
        axes[1, 0].legend()
        axes[1, 1].scatter(results['reference_lengths'], results['translation_lengths'], alpha=0.6)
        axes[1, 1].plot([0, max(results['reference_lengths'])], [0, max(results['reference_lengths'])], 'r--', label='Perfect match')
        axes[1, 1].set_title('Translation vs Reference Length')
        axes[1, 1].set_xlabel('Reference Length (words)')
        axes[1, 1].set_ylabel('Translation Length (words)')
        axes[1, 1].legend()
        
        if len(results['bleu_scores']) > 1:
            score_data = pd.DataFrame({
                'BLEU': results['bleu_scores'],
                'METEOR': results['meteor_scores'],
                'Word_Overlap': results['word_overlaps']
            })
            
            if ROUGE_AVAILABLE:
                score_data['ROUGE-1'] = [r['rouge1'] for r in results['rouge_scores']]
            
            corr_matrix = score_data.corr()
            sns.heatmap(corr_matrix, annot=True, ax=axes[1, 2], cmap='coolwarm', center=0)
            axes[1, 2].set_title('Score Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(results['reference_lengths'], results['bleu_scores'], alpha=0.6)
        axes[0].set_title('BLEU Score vs Reference Length')
        axes[0].set_xlabel('Reference Length (words)')
        axes[0].set_ylabel('BLEU Score')
        axes[1].scatter(results['reference_lengths'], results['word_overlaps'], alpha=0.6)
        axes[1].set_title('Word Overlap vs Reference Length')
        axes[1].set_xlabel('Reference Length (words)')
        axes[1].set_ylabel('Word Overlap')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_vs_length.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved successfully!")
    
    def save_detailed_results(self, results: Dict, aggregates: Dict, output_dir: Path):
        print("Saving detailed results...")
        translations_df = pd.DataFrame(results['translations'])
        translations_df.to_csv(output_dir / 'detailed_translations.csv', index=False, encoding='utf-8')

        with open(output_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(aggregates, f, ensure_ascii=False, indent=2)
        
        summary_report = self.create_summary_report(aggregates)
        with open(output_dir / 'evaluation_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print("Results saved successfully!")
    
    def create_summary_report(self, aggregates: Dict) -> str:
        report = []
        report.append("BRAJ-HINDI TRANSLATION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("BLEU SCORES:")
        report.append(f"  Mean BLEU: {aggregates['bleu']['mean']:.4f} ± {aggregates['bleu']['std']:.4f}")
        report.append(f"  Median BLEU: {aggregates['bleu']['median']:.4f}")
        report.append(f"  Range: {aggregates['bleu']['min']:.4f} - {aggregates['bleu']['max']:.4f}")
        report.append("")
        
        if ROUGE_AVAILABLE:
            report.append("ROUGE SCORES:")
            report.append(f"  ROUGE-1: {aggregates['rouge1']['mean']:.4f} ± {aggregates['rouge1']['std']:.4f}")
            report.append(f"  ROUGE-2: {aggregates['rouge2']['mean']:.4f} ± {aggregates['rouge2']['std']:.4f}")
            report.append(f"  ROUGE-L: {aggregates['rougeL']['mean']:.4f} ± {aggregates['rougeL']['std']:.4f}")
            report.append("")
        
        report.append("METEOR SCORES:")
        report.append(f"  Mean METEOR: {aggregates['meteor']['mean']:.4f} ± {aggregates['meteor']['std']:.4f}")
        report.append("")

        report.append("OTHER METRICS:")
        report.append(f"  Exact Match Accuracy: {aggregates['exact_match_accuracy']:.4f}")
        report.append(f"  Word Overlap (Jaccard): {aggregates['word_overlap']['mean']:.4f} ± {aggregates['word_overlap']['std']:.4f}")
        report.append("")
        
        report.append("LENGTH STATISTICS:")
        report.append(f"  Avg Prediction Length: {aggregates['length_stats']['avg_prediction_length']:.1f} words")
        report.append(f"  Avg Reference Length: {aggregates['length_stats']['avg_reference_length']:.1f} words")
        report.append(f"  Length Ratio: {aggregates['length_stats']['length_ratio']:.3f}")
        report.append("")
        
        report.append("QUALITY ASSESSMENT:")
        bleu_mean = aggregates['bleu']['mean']
        if bleu_mean >= 0.4:
            quality = "Excellent"
        elif bleu_mean >= 0.3:
            quality = "Good"
        elif bleu_mean >= 0.2:
            quality = "Fair"
        else:
            quality = "Needs Improvement"
        
        report.append(f"  Overall Quality: {quality} (BLEU: {bleu_mean:.4f})")
        report.append("")
        
        return "\n".join(report)
    
    def run_full_evaluation(self, max_samples: int = None):
        print("Starting comprehensive model evaluation...")
        print("=" * 60)
        self.load_model_and_data()
        results = self.evaluate_model(max_samples)
        aggregates = self.calculate_aggregate_metrics(results)
        eval_output_dir = self.model_dir / "evaluation_results"
        eval_output_dir.mkdir(exist_ok=True)
        self.create_evaluation_plots(results, eval_output_dir)
        self.save_detailed_results(results, aggregates, eval_output_dir)
        print("\n" + self.create_summary_report(aggregates))
        print(f"\nEvaluation completed!")
        print(f"Results saved in: {eval_output_dir}")
        
        return results, aggregates

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Braj-Hindi Translation Model")
    parser.add_argument("--model_dir", type=str, default="BrajHindiModelV3", help="Directory containing the trained model")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    args = parser.parse_args()
    
    try:
        evaluator = TranslationEvaluator(args.model_dir)
        evaluator.run_full_evaluation(args.max_samples)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
