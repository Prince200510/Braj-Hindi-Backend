import os
import sys

def check_requirements():
    required_files = [
        "model.py",
        "train.py", 
        "translate.py",
        "dataset.csv"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("All required files present")
    return True

def check_trained_model():
    model_files = [
        "best_model_weights.weights.h5",
        "model_metadata.pkl",
        "braj_tokenizer.model",
        "hindi_tokenizer.model"
    ]
    
    missing = [f for f in model_files if not os.path.exists(f)]
    if missing:
        print("Model not trained yet. Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun 'python train.py' first to train the model!")
        return False
    
    print("Trained model found")
    return True

def demo_translations():
    """Demonstrate translation capabilities"""
    try:
        from translate import BrajHindiTranslator
        
        print("\nLoading translator...")
        translator = BrajHindiTranslator(
            model_weights_path="final_model_weights.weights.h5",
            metadata_path="model_metadata.pkl", 
            braj_tokenizer_path="braj_tokenizer.model",
            hindi_tokenizer_path="hindi_tokenizer.model"
        )
        
        print("Translator loaded successfully!\n")
        
        test_cases = [
            ("हौं", "word"),
            ("पांव एड़ी", "auto"), 
            ("करब", "word"),
            ("जाब", "word"),
            ("हौं जाब", "sentence"),
            ("तुम आवब", "sentence"),
            ("क्रिसमस की मंगलकामना दीवाली की मंगलकामना ", "auto"),
            ("दीवाली की म ईद मुबारक", "auto"),
            ("मंगलकामना", "auto")
        ]
        
        print("Demo Translations:")
        print("=" * 50)
        
        for braj_text, mode in test_cases:
            try:
                hindi_text = translator.translate(braj_text, mode)
                print(f"Braj: '{braj_text}' → Hindi: '{hindi_text}' [{mode}]")
            except Exception as e:
                print(f"Braj: '{braj_text}' → Error: {e} [{mode}]")
        
        print("\nDemo completed")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Demo error: {e}")

def main():
    print("Braj-to-Hindi NMT System Demo")
    print("=" * 40)
    
    if not check_requirements():
        return
    
    if not check_trained_model():
        print("\nTo train the model, run:")
        print("   python train.py")
        print("\nTraining takes approximately 15-30 minutes")
        return

    demo_translations()
    
    print("\nNext steps:")
    print("  - Interactive mode: python translate.py --interactive")
    print("  - Single translation: python translate.py --input 'हौं जाब'")
    print("  - Help: python translate.py --help")

if __name__ == "__main__":
    main()
