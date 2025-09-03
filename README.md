# usb-mic-fasterwhisper
A real-time speech-to-text system that streams audio from a USB microphone to faster-whisper, applies multi-layer filtering, and outputs clean live subtitles via a web interface or SRT subtitle files.

✨ Features

🎙️ USB microphone input with real-time streaming

⚡ Low-latency transcription using faster-whisper

🧹 Multi-layer filtering:

Minimum length & word count

Similarity suppression (avoid near-duplicates)

Hallucination filter (CSV list of phrases)

🎚️ Voice Activity Detection (VAD) with Silero (preferred) or energy-based fallback

📊 Performance monitor: latency, skipped chunks, processing lag

📺 Web subtitle overlay (OBS-ready via Browser Source)

📝 SRT file generation for recordings

🔧 Requirements

Python 3.10+

Recommended: GPU with CUDA (CPU also supported)