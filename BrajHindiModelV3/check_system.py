def check_dependencies():
    results = {}
    dependencies = [
        ('torch', 'PyTorch'),
        ('sentencepiece', 'SentencePiece'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('nltk', 'NLTK'),
        ('tqdm', 'TQDM')
    ]
    
    print("Checking Braj-Hindi Translation System Dependencies")
    print("=" * 60)
    
    all_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"{name}: Available")
            results[module] = True
        except ImportError:
            print(f"{name}: Not installed")
            results[module] = False
            all_available = False
    
    print("\n" + "=" * 60)
    
    if all_available:
        print("All dependencies are available!")
        print("System is ready to run")
        
        if results.get('torch', False):
            try:
                import torch
                print(f"\nPyTorch Details:")
                print(f"Version: {torch.__version__}")
                print(f"CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA Device: {torch.cuda.get_device_name()}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except:
                pass
        
        return True
    else:
        print("Some dependencies are missing!")
        print("\nTo install missing dependencies:")
        print("pip install torch torchvision torchaudio sentencepiece pandas numpy matplotlib scikit-learn nltk tqdm")
        return False

def check_data_files():
    """Check if required data files exist"""
    from pathlib import Path
    
    print("\nmaking Data Files")
    print("=" * 30)
    
    base_dir = Path("../")
    required_files = [
        (base_dir / "word.csv", "Word pairs dataset"),
        (base_dir / "Dataset_v1.csv", "Sentence pairs dataset")
    ]
    
    all_files_exist = True
    
    for file_path, description in required_files:
        if file_path.exists():
            print(f"{description}: Found at {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                print(f"Contains {lines} lines")
            except:
                print(f"File exists but couldn't read line count")
        else:
            print(f"{description}: Missing at {file_path}")
            all_files_exist = False
    
    return all_files_exist

def check_system_readiness():
    print("BRAJ-HINDI TRANSLATION SYSTEM STATUS CHECK")
    print("=" * 70)
    
    deps_ok = check_dependencies()
    data_ok = check_data_files()
    
    print("\n" + "=" * 70)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 70)
    
    if deps_ok and data_ok:
        print("STATUS: READY TO TRAIN")
        print("All dependencies installed")
        print("All data files present")
        print("\nNext steps:")
        print("   1. Run: python braj_hindi_transformer.py (to train)")
        print("   2. Run: python inference.py --interactive (to test)")
        print("   3. Run: python evaluation.py (to evaluate)")
        
    elif deps_ok and not data_ok:
        print("STATUS: DEPENDENCIES OK, DATA MISSING")
        print("All dependencies installed")
        print("Some data files missing")
        print("\nAction needed:")
        print("   - Ensure word.csv and Dataset_v1.csv are in parent directory")
        
    elif not deps_ok and data_ok:
        print("STATUS: DATA OK, DEPENDENCIES MISSING")
        print("Some dependencies missing")
        print("All data files present")
        print("\nAction needed:")
        print("   - Install missing Python packages")
        
    else:
        print("STATUS: NOT READY")
        print("Dependencies missing")
        print("Data files missing")
        print("\nAction needed:")
        print("   1. Install missing Python packages")
        print("   2. Ensure data files are available")

if __name__ == "__main__":
    check_system_readiness()
