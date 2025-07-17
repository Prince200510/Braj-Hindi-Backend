import os
import sys
import subprocess

def check_requirements():
    required_files = [
        "model_metadata.pkl",
        "braj_tokenizer.model", 
        "hindi_tokenizer.model"
    ]
    model_files = ["best_model_weights.weights.h5", "final_model_weights.weights.h5"]
    model_exists = any(os.path.exists(f) for f in model_files)
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files or not model_exists:
        print("Missing required files:")
        if missing_files:
            for file in missing_files:
                print(f"  - {file}")
        if not model_exists:
            print(f"  - Model weights (one of: {', '.join(model_files)})")
        print("\nRun 'python train.py' first to train the model!")
        return False
    
    return True

def install_requirements():
    try:
        import fastapi
        import uvicorn
        import tensorflow
        import sentencepiece
        print("All requirements are installed")
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install requirements")
            return False

def main():
    print("Braj-Hindi Translation API Server")
    print("=" * 50)
    
    if not os.path.exists("translate.py"):
        print("Please run this script from the backend directory")
        return
    
    if not check_requirements():
        return

    if not install_requirements():
        return
    
    print("\nStarting server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
