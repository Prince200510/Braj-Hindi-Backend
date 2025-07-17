import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download('punkt')

def read_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    references = read_lines('references.txt') 
    predictions = read_lines('predictions.txt')  

    assert len(references) == len(predictions), "Mismatch in number of lines."

    references_tokenized = [[nltk.word_tokenize(ref)] for ref in references]
    predictions_tokenized = [nltk.word_tokenize(pred) for pred in predictions]

    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references_tokenized, predictions_tokenized, smoothing_function=smoothie)

    print(f"BLEU score for Fine_tune_V4 model: {bleu_score * 100:.2f}")

if __name__ == "__main__":
    main()
