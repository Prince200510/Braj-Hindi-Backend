import subprocess
import sys
import os
from pathlib import Path
import json
import time

def install_requirements():
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "nltk>=3.8.0",
        "sacrebleu>=2.3.0"
    ]
    
    print("Installing required packages for mBART fine-tuning...")
    
    for package in requirements:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{package} installed")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    print("Installation completed!")

def check_gpu():
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("GPU not available - training will be slower on CPU")
            return False
            
    except ImportError:
        print("PyTorch not installed")
        return False

def verify_dataset():
    dataset_path = Path("Dataset_v1.csv")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        print(f"Dataset verification:")
        print(f"   File: {dataset_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        if len(df) < 1000:
            print("Small dataset - consider adding more examples for better results")
        elif len(df) >= 5000:
            print("Good dataset size for fine-tuning")
        
        print(f"   Sample entries:")
        for i in range(min(3, len(df))):
            print(f"     Braj:  {df.iloc[i]['braj_translation']}")
            print(f"     Hindi: {df.iloc[i]['hindi_sentence']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

def run_fine_tuning():
    print("Starting mBART fine-tuning process...")
    
    try:
        result = subprocess.run([sys.executable, "mbart_fine_tune.py"], 
                              capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            print("Fine-tuning completed successfully!")
            return True
        else:
            print(f"Fine-tuning failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Fine-tuning timed out (2 hours)")
        return False
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False

def test_fine_tuned_model():
    print("Testing fine-tuned model...")
    
    model_dir = Path("./fine_tuned_mbart_braj_hindi")
    
    if not model_dir.exists():
        print(f"Fine-tuned model not found: {model_dir}")
        return False
    
    try:
        result = subprocess.run([sys.executable, "mbart_inference.py", "--test"], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Model testing completed!")
            print(result.stdout)
            return True
        else:
            print(f"Model testing failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def create_evaluation_report():
    print("Creating evaluation report...")
    
    try:
        import torch
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        import pandas as pd
        
        model_dir = "./fine_tuned_mbart_braj_hindi"
        
        if not Path(model_dir).exists():
            print(f"Model directory not found: {model_dir}")
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
        model = MBartForConditionalGeneration.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        
        df = pd.read_csv("Dataset_v1.csv")
        test_samples = df.sample(n=min(50, len(df)), random_state=42)
        
        translations = []
        translation_times = []
        
        print("   Evaluating sample translations...")
        
        for _, row in test_samples.iterrows():
            braj_text = row['braj_translation']
            expected_hindi = row['hindi_sentence']

            start_time = time.time()

            input_text = f"Braj: {braj_text}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, 
                             truncation=True, padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=3, 
                                       early_stopping=True)

            predicted_hindi = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation_time = time.time() - start_time

            translations.append({
                'braj': braj_text,
                'expected_hindi': expected_hindi,
                'predicted_hindi': predicted_hindi,
                'translation_time': translation_time
            })
            translation_times.append(translation_time)
        
        avg_time = sum(translation_times) / len(translation_times)
        
        report = {
            "model_info": {
                "model_path": model_dir,
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(device),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance": {
                "avg_translation_time": avg_time,
                "translations_per_second": 1 / avg_time,
                "test_samples": len(test_samples),
                "total_dataset_size": len(df)
            },
            "sample_translations": translations[:10]
        }
        
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation report saved to: evaluation_report.json")
        print(f"   Average translation time: {avg_time:.3f}s")
        print(f"   Translations per second: {1/avg_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"Error creating evaluation report: {e}")
        return False

def main():
    print("mBART Fine-tuning Pipeline for Braj-Hindi Translation")
    print("=" * 60)
    
    steps = [
        ("Install Requirements", install_requirements),
        ("Check GPU", check_gpu),
        ("Verify Dataset", verify_dataset),
        ("Run Fine-tuning", run_fine_tuning),
        ("Test Model", test_fine_tuned_model),
        ("Create Report", create_evaluation_report)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        print("-" * 40)
        
        try:
            success = step_func()
            if success:
                print(f"{step_name} completed successfully")
            else:
                print(f"{step_name} failed")
                
                if step_name in ["Run Fine-tuning", "Test Model"]:
                    print("   Continuing with next steps...")
                    continue
                elif step_name == "Check GPU":
                    print("   Continuing with CPU training...")
                    continue
                else:
                    print("   Stopping pipeline due to critical failure")
                    break
                    
        except Exception as e:
            print(f"{step_name} error: {e}")
            if step_name in ["Install Requirements", "Verify Dataset"]:
                print("   Stopping pipeline due to critical failure")
                break
    
    print(f"\nPipeline completed!")
    print(f"Check the following files:")
    print(f"   - ./fine_tuned_mbart_braj_hindi/ (fine-tuned model)")
    print(f"   - evaluation_report.json (performance report)")
    print(f"   - mbart_inference.py (run inference)")
    
    print(f"\nTo use the model:")
    print(f"   python mbart_inference.py --interactive")
    print(f"   python mbart_inference.py --test")
    print(f"   python mbart_inference.py --translate 'your text here'")

if __name__ == "__main__":
    main()
