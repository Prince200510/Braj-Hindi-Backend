import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time
import json
import os
from pathlib import Path

class FastBrajTranslator:
    def __init__(self, model_path="./fine_tuned_mbart_braj_hindi"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_time = 0
        
    def load_model(self):
        start_time = time.time()
        print(f"Loading fine-tuned mBART model from {self.model_path}...")
        
        if not self.model_path.exists():
            print(f"Error: Model directory {self.model_path} not found!")
            print("Please run mbart_fine_tune.py first to create the fine-tuned model.")
            return False
        
        try:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.load_time = time.time() - start_time
            
            print(f"Model loaded successfully in {self.load_time:.2f}s")
            print(f"Device: {self.device}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def translate(self, braj_text, max_length=128, num_beams=3, temperature=0.8):
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return ""

        start_time = time.time()

        input_text = f"Braj: {braj_text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False,
                top_p=0.9 if temperature > 0 else None
            )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation_time = time.time() - start_time

        return {
            'translation': translation.strip(),
            'time': translation_time,
            'source': braj_text
        }
    
    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("Real-time Braj to Hindi Translation")
        print("=" * 60)
        print("Commands: 'quit', 'stats', 'help', 'settings'")
        print("-" * 60)

        translation_count = 0
        total_time = 0
        settings = {'max_length': 128, 'num_beams': 3, 'temperature': 0.8}

        while True:
            try:
                user_input = input("\nBraj: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                elif user_input.lower() == 'stats':
                    avg_time = total_time / translation_count if translation_count > 0 else 0
                    print(f"\nTranslation Statistics:")
                    print(f"   Total translations: {translation_count}")
                    print(f"   Average time: {avg_time:.3f}s")
                    print(f"   Total time: {total_time:.2f}s")
                    print(f"   Model load time: {self.load_time:.2f}s")
                    continue

                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("   'quit' - Exit the translator")
                    print("   'stats' - Show translation statistics")
                    print("   'settings' - Adjust translation parameters")
                    print("   'help' - Show this help message")
                    continue

                elif user_input.lower() == 'settings':
                    print(f"\nCurrent Settings:")
                    print(f"   Max length: {settings['max_length']}")
                    print(f"   Num beams: {settings['num_beams']}")
                    print(f"   Temperature: {settings['temperature']}")

                    try:
                        new_beams = input("Enter new num_beams (1-5, current: {}): ".format(settings['num_beams']))
                        if new_beams.strip():
                            settings['num_beams'] = max(1, min(5, int(new_beams)))

                        new_temp = input("Enter new temperature (0.1-2.0, current: {}): ".format(settings['temperature']))
                        if new_temp.strip():
                            settings['temperature'] = max(0.1, min(2.0, float(new_temp)))

                        print("Settings updated!")
                    except ValueError:
                        print("Invalid input. Settings unchanged.")
                    continue

                if user_input:
                    result = self.translate(
                        user_input,
                        max_length=settings['max_length'],
                        num_beams=settings['num_beams'],
                        temperature=settings['temperature']
                    )

                    total_time += result['time']
                    translation_count += 1

                    print(f"Hindi: {result['translation']}")
                    print(f"⏱Time: {result['time']:.3f}s")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print(f"\nSession Summary:")
        print(f"   Translations: {translation_count}")
        if translation_count > 0:
            print(f"   Average time: {total_time / translation_count:.3f}s")
        print("Thank you for using the translator!")

def batch_translate_file(input_file, output_file, model_path="./fine_tuned_mbart_braj_hindi"):
    translator = FastBrajTranslator(model_path)
    
    if not translator.load_model():
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Translating {len(lines)} lines from {input_file}...")

        results = []
        total_time = 0

        for i, line in enumerate(lines, 1):
            result = translator.translate(line)
            results.append({
                'source_braj': line,
                'translation_hindi': result['translation'],
                'time': result['time']
            })
            total_time += result['time']

            if i % 10 == 0:
                print(f"   Processed {i}/{len(lines)} lines...")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Batch translation completed!")
        print(f"   Output saved to: {output_file}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per translation: {total_time/len(lines):.3f}s")

        return True

    except Exception as e:
        print(f"Error during batch translation: {e}")
        return False

def test_model_performance():
    model_path = "./fine_tuned_mbart_braj_hindi"
    
    print("Testing Fine-tuned mBART Model Performance")
    print("=" * 50)
    
    translator = FastBrajTranslator(model_path)
    
    if not translator.load_model():
        print("Cannot test - model not found or failed to load")
        return
    
    test_sentences = [
        "तोहे देख के मन हरषायो",
        "हमारो घर बड़ो है",
        "कित किताब पढ़त है",
        "म्यां बजार जात हौं",
        "तुम कइसे हो",
        "मोहे ई बात समझ मा आयी",
        "गांव के लोग अच्छे हैं",
        "बच्चा खेलत है",
        "आज मौसम बढ़िया है",
        "हम कल मिलबै"
    ]

    print(f"Testing {len(test_sentences)} sample translations:")
    print("-" * 50)

    total_time = 0

    for i, sentence in enumerate(test_sentences, 1):
        result = translator.translate(sentence)
        total_time += result['time']

        print(f"{i:2d}. Braj: {sentence}")
        print(f"    Hindi:  {result['translation']}")
        print(f"    Time:  {result['time']:.3f}s")
        print()

    avg_time = total_time / len(test_sentences)

    print("Performance Summary:")
    print(f"   Total test time: {total_time:.2f}s")
    print(f"   Average per translation: {avg_time:.3f}s")
    print(f"   Model load time: {translator.load_time:.2f}s")
    print(f"   Translations per second: {1/avg_time:.1f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuned mBART Braj-to-Hindi Translator")
    parser.add_argument('--model-path', default='./fine_tuned_mbart_braj_hindi', help='Path to fine-tuned model directory')
    parser.add_argument('--interactive', action='store_true', default=True,help='Start interactive translation mode')
    parser.add_argument('--test', action='store_true',help='Run performance test')
    parser.add_argument('--batch', nargs=2, metavar=('INPUT', 'OUTPUT'),help='Batch translate file: INPUT OUTPUT')
    parser.add_argument('--translate', help='Translate single sentence')
    args = parser.parse_args()
    
    if args.test:
        test_model_performance()
    elif args.batch:
        batch_translate_file(args.batch[0], args.batch[1], args.model_path)
    elif args.translate:
        translator = FastBrajTranslator(args.model_path)
        if translator.load_model():
            result = translator.translate(args.translate)
            print(f"Hindi: {args.translate}")
            print(f"Braj:  {result['translation']}")
    else:
        translator = FastBrajTranslator(args.model_path)
        if translator.load_model():
            translator.interactive_mode()

if __name__ == "__main__":
    main()
