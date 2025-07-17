# Braj-Hindi Translation API Backend

A high-performance FastAPI server for neural machine translation from Braj to Hindi using TensorFlow.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Ensure Model Files Exist

Make sure you have the following files in the backend directory:

- `model_metadata.pkl`
- `braj_tokenizer.model`
- `hindi_tokenizer.model`
- `best_model_weights.weights.h5` OR `final_model_weights.weights.h5`

If these files don't exist, run:

```bash
python train.py
```

### 3. Start the Server

**Option A: Using the startup script**

```bash
python start_server.py
```

**Option B: Using uvicorn directly**

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Windows batch file**

```bash
start_server.bat
```

### 4. Verify Server is Running

- Server: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Translation

```
POST /translate
```

**Request Body:**

```json
{
  "text": "हौं जाब",
  "mode": "auto"
}
```

**Response:**

```json
{
  "original_text": "हौं जाब",
  "translated_text": "मैं जाऊंगा",
  "mode": "auto",
  "processing_time": 0.123
}
```

**Modes:**

- `auto`: Automatically choose best translation strategy
- `word`: Word-by-word translation
- `sentence`: Full sentence translation

### Health Check

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Translation model is loaded and ready"
}
```

### Model Information

```
GET /model/info
```

**Response:**

```json
{
  "model_loaded": true,
  "max_sequence_length": 50,
  "input_vocab_size": 8000,
  "target_vocab_size": 8000,
  "supported_modes": ["auto", "word", "sentence"],
  "source_language": "Braj",
  "target_language": "Hindi"
}
```

## Features

- **High Performance**: TensorFlow-optimized neural translation
- **Fast Loading**: Model loads asynchronously on startup
- **CORS Enabled**: Ready for frontend integration
- **Error Handling**: Comprehensive error messages
- **Health Monitoring**: Built-in health checks
- **Multiple Modes**: Auto, word-level, and sentence-level translation
- **Processing Time**: Response includes translation timing

## Frontend Integration

The API is configured to work with the React frontend running on `http://localhost:3000`. The frontend automatically calls this backend for translations.

## Development

### Environment Setup

1. Python 3.8+
2. TensorFlow 2.10+
3. FastAPI and Uvicorn
4. All dependencies in `requirements.txt`

### Model Loading

The model loads automatically on server startup in a background thread, so the server starts quickly even with large models.

### Error Handling

- Model loading errors are logged
- Translation errors return appropriate HTTP status codes
- Input validation prevents malformed requests

## Production Deployment

For production deployment:

1. Use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. Configure proper CORS origins for your domain
3. Add proper logging and monitoring
4. Use environment variables for configuration

## Troubleshooting

### Model Not Loading

- Ensure all model files exist and are not corrupted
- Check Python path and dependencies
- Review server logs for detailed error messages

### Slow Performance

- Model loads in background, wait for completion
- First translation may be slower due to TensorFlow initialization
- Subsequent translations should be much faster

### Connection Issues

- Ensure server is running on port 8000
- Check firewall settings
- Verify CORS configuration matches frontend URL
