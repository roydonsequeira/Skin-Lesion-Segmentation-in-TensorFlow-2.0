# Kokoro-82M Text-to-Speech Migration Guide

## Overview
This guide covers the migration from DeepGram TTS to Kokoro-82M, an open-weight text-to-speech model.

## Changes Made

### 1. Dependencies Updated
- Added `kokoro>=0.9.4` to requirements.txt
- Added `soundfile` for audio processing

### 2. Code Changes
- **services/text2speech.py**: Completely replaced DeepGram implementation with Kokoro-82M
- **app.py**: Updated text-to-speech endpoint to remove DeepGram API key dependency

## Installation Steps

### 1. Install System Dependencies (Linux/Ubuntu)
```bash
sudo apt-get install espeak-ng
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. First Run Considerations
- The first run will download the Kokoro-82M model (~82MB)
- GPU support will be automatically detected and used if available
- CPU fallback is available if no GPU is present

## Configuration Options

### Available Voices
You can customize the voice by modifying the `voice` parameter in `services/text2speech.py`:
- `'af_bella'` (default)
- `'af_sarah'`
- `'af_nicole'`
- And other available voices from the Kokoro model

### Device Selection
The system automatically detects and uses:
- CUDA GPU if available (faster)
- CPU if no GPU is present (slower but functional)

## Testing

### 1. Test the Endpoint
```bash
curl -X POST http://localhost:5000/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of Kokoro-82M"}' \
  --output test_output.wav
```

### 2. Verify Audio Output
Play the generated `test_output.wav` file to verify quality.

## Benefits of Migration

### Advantages
- **No API Key Required**: Runs locally without external API calls
- **Cost Effective**: No per-request charges
- **Privacy**: All processing happens locally
- **Offline Capability**: Works without internet connection
- **High Quality**: Kokoro-82M provides excellent speech synthesis

### Considerations
- **Initial Setup**: Requires model download on first run
- **Resource Usage**: Uses local CPU/GPU resources
- **Model Size**: ~82MB model storage requirement

## Troubleshooting

### Common Issues
1. **espeak-ng not found**: Install system dependency
2. **CUDA out of memory**: Model will fallback to CPU automatically
3. **Import errors**: Ensure all dependencies are installed correctly

### Performance Optimization
- Use GPU when available for faster inference
- Consider batch processing for multiple requests
- Monitor memory usage for long-running applications

## Rollback Plan
If you need to rollback to DeepGram:
1. Restore the original `services/text2speech.py`
2. Restore the original `app.py` text-to-speech endpoint
3. Remove Kokoro dependencies from requirements.txt
4. Ensure DEEPGRAM_API_KEY is configured
