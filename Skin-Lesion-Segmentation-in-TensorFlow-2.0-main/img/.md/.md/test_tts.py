#!/usr/bin/env python3

from kokoro import KPipeline
import numpy as np
import soundfile as sf
import os

pipe = KPipeline(lang_code='a')  # American English
text = "Hello, this is the Kokoro eighty-two M text-to-speech model."
segments = []
for _, _, audio in pipe(text, voice='af_heart'):
    segments.append(audio)
audio = np.concatenate(segments) if len(segments) > 1 else (segments[0] if segments else np.zeros(24000))
os.makedirs("v6/AsendHealth/data", exist_ok=True)
sf.write("v6/AsendHealth/data/output.wav", audio, 24000)
print("Wrote: v6/AsendHealth/data/output.wav")