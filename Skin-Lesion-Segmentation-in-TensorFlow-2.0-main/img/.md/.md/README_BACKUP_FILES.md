# Backup Files - Deepgram Implementation (Reference Only)

This directory contains backup/reference files with the original Deepgram implementation that was replaced with OpenAI Whisper API.

## Files

### `app_deepgram_backup.py`
- Original Flask app using Deepgram API
- Uses `DEEPGRAM_API_KEY` environment variable
- Simplified audio handling (no multipart parsing)
- For reference only - do not use in production

### `services/speech2text_deepgram_backup.py`
- Original Deepgram speech-to-text implementation
- Uses Deepgram API endpoint: `https://api.deepgram.com/v1/listen`
- Authorization: `Token {API_KEY}`
- Response format: `result['results']['channels'][0]['alternatives'][0]['transcript']`
- For reference only - do not use in production

## Current Implementation

The current active implementation uses:
- **Service**: OpenAI Whisper API
- **File**: `app.py` (main Flask app)
- **STT Module**: `services/speech2text.py`
- **Environment Variable**: `OPENAI_API_KEY`

## Migration Notes

**Changed:**
- Deepgram API → OpenAI Whisper API
- `DEEPGRAM_API_KEY` → `OPENAI_API_KEY`
- Authorization: `Token {key}` → `Bearer {key}`
- Endpoint: Deepgram `/v1/listen` → OpenAI `/v1/audio/transcriptions`
- Response parsing: Different JSON structure
- Added better multipart form-data handling
- Added format detection from Content-Type headers

**Reasons for Migration:**
- Better reliability with file uploads
- Simpler API usage
- Better format support
- More reliable multipart parsing

## To Use Backup (Not Recommended)

If you need to temporarily use the Deepgram implementation:

1. Set environment variable: `DEEPGRAM_API_KEY`
2. Rename files:
   - `app_deepgram_backup.py` → `app.py`
   - `services/speech2text_deepgram_backup.py` → `services/speech2text.py`
3. Update imports in `app_deepgram_backup.py` if needed

**Note**: The backup files have simplified error handling and may not work with all Postman request formats that the current implementation supports.
