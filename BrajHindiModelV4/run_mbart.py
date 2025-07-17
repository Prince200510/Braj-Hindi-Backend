import os
import subprocess
import sys
from pathlib import Path

def quick_start():
    print("Quick Start: mBART Fine-tuning for Braj-to-Hindi Translation")
    print("=" * 60)
    
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    
    required_files = ["Dataset_v1.csv", "mbart_fine_tune.py", "mbart_inference.py"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"Found: {file}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    print("\nChoose an option:")
    print("1. Full pipeline (install + fine-tune + test)")
    print("2. Fine-tune only (if dependencies already installed)")
    print("3. Test existing model")
    print("4. Interactive translation")
    print("5. Install dependencies only")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning full pipeline...")
            subprocess.run([sys.executable, "setup_mbart.py"])
            
        elif choice == "2":
            print("\nStarting fine-tuning...")
            subprocess.run([sys.executable, "mbart_fine_tune.py"])
            
        elif choice == "3":
            print("\nTesting existing model...")
            subprocess.run([sys.executable, "mbart_inference.py", "--test"])
            
        elif choice == "4":
            print("\nStarting interactive translation...")
            subprocess.run([sys.executable, "mbart_inference.py", "--interactive"])
            
        elif choice == "5":
            print("\nInstalling dependencies...")
            packages = [
                "torch", "transformers", "datasets", "pandas", 
                "numpy", "scikit-learn", "tqdm", "accelerate", "evaluate"
            ]
            for package in packages:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
            print("Dependencies installed!")
            
        else:
            print("Invalid choice")
            return False
            
        return True
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def show_help():
    print("mBART Fine-tuning Help")
    print("=" * 30)
    print()
    print("Files in this project:")
    print("   • mbart_fine_tune.py    - Main fine-tuning script")
    print("   • mbart_inference.py    - Inference and testing")
    print("   • setup_mbart.py        - Full pipeline setup")
    print("   • Dataset_v1.csv        - Training data (7000+ pairs)")
    print()
    print("Quick commands:")
    print("   python setup_mbart.py           # Full pipeline")
    print("   python mbart_fine_tune.py       # Fine-tune only")
    print("   python mbart_inference.py       # Interactive mode")
    print("   python mbart_inference.py --test # Test performance")
    print()
    print("Expected results:")
    print("   • Model: ~600M parameters (mBART-large)")
    print("   • Training time: 2-4 hours (GPU) / 8-12 hours (CPU)")
    print("   • Inference speed: ~0.1-0.5s per translation")
    print("   • Memory usage: 2-4GB GPU / 4-8GB RAM")
    print()
    print("System requirements:")
    print("   • Python 3.8+")
    print("   • PyTorch 2.0+")
    print("   • 8GB+ RAM (16GB+ recommended)")
    print("   • CUDA GPU (recommended for training)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        quick_start()
