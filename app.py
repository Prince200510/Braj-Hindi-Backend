import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Optional, List
import asyncio
import threading
import time
import pandas as pd
from translate import BrajHindiTranslator
sys.path.append(os.path.join(os.path.dirname(__file__), 'BrajHindiModelV4'))
from BrajHindiModelV4.mbart_inference import FastBrajTranslator

sys.path.append(os.path.join(os.path.dirname(__file__), 'BrajHindiModelV3'))
try:
    from inference import InferenceInterface
    V3_AVAILABLE = True
except ImportError as e:
    print(f"V3 model not available: {e}")
    V3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Braj-Hindi Translation API",
    description="High-performance neural machine translation from Braj to Hindi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = None
translator_loading = False
translator_loaded = False
v3_translator = None
v3_translator_loaded = False
v4_translator = None
v4_translator_loaded = False

class TranslationRequest(BaseModel):
    text: str
    mode: Optional[str] = "auto"  
    model: Optional[str] = "v4"  

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    mode: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

class DatasetEntry(BaseModel):
    id: int
    hindi_sentence: str
    braj_translation: str

class DatasetResponse(BaseModel):
    total_entries: int
    entries: List[DatasetEntry]
    page: int
    per_page: int
    total_pages: int

def load_translator():
    global translator, translator_loading, translator_loaded
    
    if translator_loaded or translator_loading:
        return
    
    translator_loading = True
    logger.info("Starting to load translation model...")
    
    try:
        required_files = [
            "model_metadata.pkl",
            "braj_tokenizer.model", 
            "hindi_tokenizer.model"
        ]
        
        model_path = None
        if os.path.exists("best_model_weights.weights.h5"):
            model_path = "best_model_weights.weights.h5"
        elif os.path.exists("final_model_weights.weights.h5"):
            model_path = "final_model_weights.weights.h5"
        else:
            raise FileNotFoundError("No model weights found")
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        translator = BrajHindiTranslator(
            model_weights_path=model_path,
            metadata_path="model_metadata.pkl",
            braj_tokenizer_path="braj_tokenizer.model",
            hindi_tokenizer_path="hindi_tokenizer.model"
        )
        
        translator_loaded = True
        translator_loading = False
        logger.info("Translation model loaded successfully!")
        
    except Exception as e:
        translator_loading = False
        logger.error(f"Failed to load translation model: {e}")
        raise

def load_v3_translator():
    global v3_translator, v3_translator_loaded
    
    if not V3_AVAILABLE:
        logger.warning("V3 model is not available")
        return
    
    if v3_translator_loaded:
        return
    
    try:
        logger.info("Loading V3 translation model...")
        v3_model_dir = os.path.join(os.path.dirname(__file__), 'BrajHindiModelV3', 'BrajHindiModelV3')
        
        if not os.path.exists(v3_model_dir):
            logger.error(f"V3 model directory not found: {v3_model_dir}")
            return
        
        v3_translator = InferenceInterface(v3_model_dir)
        v3_translator_loaded = True
        logger.info("V3 Translation model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load V3 translation model: {e}")
        v3_translator_loaded = False

def load_v4_translator():
    global v4_translator, v4_translator_loaded
    if v4_translator_loaded:
        return
    try:
        v4_translator = FastBrajTranslator(os.path.join(os.path.dirname(__file__), 'BrajHindiModelV4', 'fine_tuned_mbart_braj_hindi'))
        loaded = v4_translator.load_model()
        v4_translator_loaded = loaded
        if loaded:
            logger.info("V4 Fine-tuned mBART model loaded successfully!")
        else:
            logger.error("Failed to load V4 Fine-tuned mBART model!")
    except Exception as e:
        v4_translator_loaded = False
        logger.error(f"Exception loading V4 model: {e}")

@app.on_event("startup")
async def startup_event():
    def load_in_thread():
        try:
            load_v4_translator()
            load_translator()
            load_v3_translator()
        except Exception as e:
            logger.error(f"Failed to load models on startup: {e}")
    
    thread = threading.Thread(target=load_in_thread)
    thread.daemon = True
    thread.start()

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="running",
        model_loaded=translator_loaded,
        message="Braj-Hindi Translation API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    global translator_loaded, translator_loading, v3_translator_loaded
    
    if translator_loaded:
        status = "healthy"
        message = f"Translation model is loaded and ready. V3 model: {'loaded' if v3_translator_loaded else 'not loaded'}"
    elif translator_loading:
        status = "loading"
        message = "Translation model is currently loading..."
    else:
        status = "not_ready"
        message = "Translation model is not loaded"
    
    return HealthResponse(
        status=status,
        model_loaded=translator_loaded,
        message=message
    )

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    global translator, translator_loaded, v3_translator, v3_translator_loaded, v4_translator, v4_translator_loaded

    model_choice = request.model or "v4"
    if model_choice not in ["v4", "current", "v3"]:
        raise HTTPException(status_code=400, detail="Invalid model. Must be 'v4', 'current', or 'v3'")

    if model_choice == "v4":
        if not v4_translator_loaded or v4_translator is None:
            raise HTTPException(
                status_code=503,
                detail="V4 translation model is not available. Please check server logs or use another model."
            )
    elif model_choice == "current":
        if not translator_loaded or translator is None:
            if translator_loading:
                raise HTTPException(
                    status_code=503,
                    detail="Current translation model is still loading. Please try again in a few moments."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Current translation model is not available. Please check server logs."
                )
    elif model_choice == "v3":
        if not v3_translator_loaded or v3_translator is None:
            raise HTTPException(
                status_code=503,
                detail="V3 translation model is not available. Please check server logs or use another model."
            )

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 500:
        raise HTTPException(status_code=400, detail="Text too long. Maximum 500 characters allowed.")

    valid_modes = ["auto", "word", "sentence"]
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    try:
        start_time = time.time()
        if model_choice == "v4":
            result = v4_translator.translate(request.text.strip())
            translated_text = result['translation']
        elif model_choice == "current":
            translated_text = translator.translate(
                text=request.text.strip(),
                mode=request.mode,
                verbose=False
            )
        elif model_choice == "v3":
            result = v3_translator.translate(
                braj_text=request.text.strip(),
                use_semantic_fallback=True
            )
            translated_text = result['recommended']

        processing_time = time.time() - start_time
        logger.info(f"Translation ({model_choice}) completed in {processing_time:.3f}s: '{request.text}' -> '{translated_text}'")
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            mode=request.mode,
            processing_time=round(processing_time, 3)
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/models")
async def get_available_models():
    """Get information about available translation models"""
    global translator_loaded, v3_translator_loaded, v4_translator_loaded
    models = {
        "v4": {
            "name": "Fine_tune_v4 (mBART)",
            "description": "Fine-tuned mBART Braj-Hindi model (highest accuracy, first priority)",
            "available": v4_translator_loaded,
            "type": "mbart"
        },
        "current": {
            "name": "Current Model (TensorFlow)",
            "description": "Standard transformer model with comprehensive dataset",
            "available": translator_loaded,
            "type": "tensorflow"
        },
        "v3": {
            "name": "BrajHindi Model V3 (PyTorch)",
            "description": "Enhanced V3 model with improved accuracy and performance",
            "available": v3_translator_loaded,
            "type": "pytorch"
        }
    }
    return {
        "models": models,
        "default": "v4"
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    global translator, translator_loaded
    
    if not translator_loaded or translator is None:
        raise HTTPException(status_code=503, detail="Translation model is not loaded")
    
    try:
        info = {
            "model_loaded": translator_loaded,
            "max_sequence_length": translator.max_seq_len,
            "input_vocab_size": translator.input_vocab_size,
            "target_vocab_size": translator.target_vocab_size,
            "supported_modes": ["auto", "word", "sentence"],
            "source_language": "Braj",
            "target_language": "Hindi"
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/dataset", response_model=DatasetResponse)
async def get_dataset(page: int = 1, per_page: int = 50, search: str = ""):
    try:
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")
        if per_page < 1 or per_page > 100:
            raise HTTPException(status_code=400, detail="per_page must be between 1 and 100")
    
        csv_path = "Dataset_v1.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        df = pd.read_csv(csv_path)
        
        if search and search.strip():
            search_term = search.strip().lower()
            mask = (
                df['hindi_sentence'].str.lower().str.contains(search_term, na=False) |
                df['braj_translation'].str.lower().str.contains(search_term, na=False)
            )
            df = df[mask]
        
        total_entries = len(df)
        total_pages = (total_entries + per_page - 1) // per_page if total_entries > 0 else 1
        offset = (page - 1) * per_page
        page_data = df.iloc[offset:offset + per_page]
        
        entries = []
        for idx, row in page_data.iterrows():
            entries.append(DatasetEntry(
                id=idx + 1,
                hindi_sentence=row['hindi_sentence'],
                braj_translation=row['braj_translation']
            ))
        
        return DatasetResponse(
            total_entries=total_entries,
            entries=entries,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
