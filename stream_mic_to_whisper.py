import asyncio
import sounddevice as sd
import json
import numpy as np
import time
import logging
import signal
import sys
import queue
import threading
import io
import os
from typing import Optional
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import webbrowser

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    print("❌ faster-whisper not installed. Install with: pip install faster-whisper")
    FASTER_WHISPER_AVAILABLE = False

# Try to import Flask and SocketIO for web interface
try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("❌ Flask or flask-socketio not installed. Install with: pip install flask flask-socketio")
    FLASK_AVAILABLE = False

# Try to import VAD dependencies
try:
    import torch
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
try:
    import ahocorasick
    import pandas as pd
    import pickle
    import hashlib
    HALLUCINATION_DETECTION_AVAILABLE = True
except ImportError:
    print("❌ Hallucination detection not available. Install with: pip install pyahocorasick pandas")
    HALLUCINATION_DETECTION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "usb_mic_config.json"
HALLUCINATION_CSV = "hallucination_phrases.csv"
SRT_OUTPUT_FILE = "live_transcription.srt"  # Output SRT file

SAMPLE_RATE = 16000
BUFFER_SIZE = 1024
CHANNELS = 1
MAX_OVERFLOW_COUNT = 3

# Audio buffering settings
CHUNK_DURATION = 3.0  # Process audio in 3-second chunks
MIN_CHUNK_DURATION = 1.5  # Minimum chunk size

# Subtitle filtering settings
MIN_TEXT_LENGTH = 10  # Minimum characters to display subtitle
MIN_WORD_COUNT = 3    # Minimum words to display subtitle
SIMILARITY_THRESHOLD = 0.6  # Skip if too similar to previous subtitle (0.0-1.0)
MIN_DISPLAY_INTERVAL = 2.0  # Minimum seconds between subtitle displays
MAX_REPETITION_TIME = 10.0  # Seconds to remember previous subtitles for deduplication

# Web server settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# Subtitle HTML template
SUBTITLE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Subtitles</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0);  /* Fully transparent for OBS */
            font-family: 'Arial', sans-serif;
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            overflow: hidden;
        }
        
        .subtitle-container {
            max-height: 200px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }
        
        .subtitle-line {
            font-size: 24px;
            line-height: 1.4;
            text-align: center;
            margin: 5px 0;
            padding: 10px 20px;
            background-color: rgba(0, 0, 0, 0.9);
            border-radius: 5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.5s ease, transform 0.5s ease;
            word-wrap: break-word;
        }
        
        .subtitle-line.show {
            opacity: 1;
            transform: translateY(0);
        }
        
        .subtitle-line.fade-out {
            opacity: 0.6;
            transform: translateY(-10px);
        }
        
        .status {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 14px;
            color: #00ff00;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px 10px;
            border-radius: 3px;
        }
        
        .confidence {
            font-size: 18px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="status" id="status">Connecting...</div>
    <div class="subtitle-container" id="subtitles">
        <div class="subtitle-line show">Waiting for audio...</div>
    </div>

    <script>
        const socket = io();
        const subtitlesContainer = document.getElementById('subtitles');
        const statusDiv = document.getElementById('status');
        const maxLines = 1;
        
        socket.on('connect', function() {
            console.log('Connected to subtitle server');
            statusDiv.textContent = 'Connected';
            statusDiv.style.color = '#00ff00';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from subtitle server');
            statusDiv.textContent = 'Disconnected';
            statusDiv.style.color = '#ff0000';
        });
        
        socket.on('subtitle', function(data) {
            console.log('Received subtitle:', data);
            addSubtitle(data.text, data.confidence || 0);
        });
        
        socket.on('clear', function() {
            clearSubtitles();
        });
        
        function addSubtitle(text, confidence) {
            // Create new subtitle line
            const subtitleLine = document.createElement('div');
            subtitleLine.className = 'subtitle-line';
            
            // Add confidence indicator if available
            const confidenceText = confidence > 0 ? 
                `<span class="confidence">[${(confidence * 100).toFixed(0)}%]</span> ` : '';
            
            subtitleLine.innerHTML = text;
            
            // Add to container
            subtitlesContainer.appendChild(subtitleLine);
            
            // Trigger animation
            setTimeout(() => {
                subtitleLine.classList.add('show');
            }, 50);
            
            // Remove old lines if we have too many
            const lines = subtitlesContainer.querySelectorAll('.subtitle-line');
            if (lines.length > maxLines) {
                const oldestLine = lines[0];
                oldestLine.classList.add('fade-out');
                setTimeout(() => {
                    if (oldestLine.parentNode) {
                        oldestLine.parentNode.removeChild(oldestLine);
                    }
                }, 500);
            }
        }
        
        function clearSubtitles() {
            subtitlesContainer.innerHTML = '<div class="subtitle-line show">Waiting for audio...</div>';
        }
        
        // Auto-clear old subtitles after 30 seconds of inactivity
        let lastSubtitleTime = Date.now();
        socket.on('subtitle', function() {
            lastSubtitleTime = Date.now();
        });
        
        setInterval(() => {
            if (Date.now() - lastSubtitleTime > 30000) {
                const lines = subtitlesContainer.querySelectorAll('.subtitle-line');
                if (lines.length > 1) {
                    clearSubtitles();
                }
            }
        }, 5000);
    </script>
</body>
</html>
'''

# Web server settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

class HallucinationDetector:
    """Detects and filters out hallucinated phrases using Aho-Corasick algorithm."""
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = HALLUCINATION_CSV
        self.csv_path = csv_path
        self.automaton = None
        self.phrase_count = 0
        
    def load_phrases(self) -> bool:
        """Load hallucination phrases from CSV file."""
        if not HALLUCINATION_DETECTION_AVAILABLE:
            logger.warning("Hallucination detection not available, phrases will not be filtered")
            return False
            
        if not os.path.exists(self.csv_path):
            logger.warning(f"Hallucination CSV file not found: {self.csv_path}")
            logger.info("Continuing without hallucination filtering")
            return False
        
        try:
            # Load CSV with proper handling of quotes and commas
            df = pd.read_csv(self.csv_path, 
                           encoding="utf-8-sig",
                           quotechar='"',
                           skipinitialspace=True,
                           na_filter=False).dropna()
            df.columns = df.columns.str.strip()
            
            if 'Phrase' not in df.columns:
                logger.error(f"CSV must contain a 'Phrase' column. Found: {df.columns.tolist()}")
                return False
            
            # Process phrases
            phrases = (df['Phrase']
                      .astype(str)
                      .str.strip()
                      .str.lower()
                      .drop_duplicates())
            
            # Remove empty phrases
            phrases = phrases[phrases.str.len() > 0]
            
            if phrases.empty:
                logger.warning(f"No valid phrases found in {self.csv_path}")
                return False
            
            # Build Aho-Corasick automaton
            self.automaton = ahocorasick.Automaton()
            for phrase in phrases:
                self.automaton.add_word(phrase, phrase)
            
            self.automaton.make_automaton()
            self.phrase_count = len(phrases)
            
            logger.info(f"Loaded {self.phrase_count} hallucination phrases for filtering")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hallucination phrases: {e}")
            return False
    
    def is_hallucination(self, text: str) -> tuple[bool, str]:
        """
        Check if text contains hallucinated phrases.
        
        Returns:
            (is_hallucination: bool, matched_phrase: str)
        """
        if not self.automaton:
            return False, ""
        
        text_lower = text.lower().strip()
        
        # Find all matches
        matches = list(self.automaton.iter(text_lower))
        
        if matches:
            # Return the longest matched phrase
            longest_match = max(matches, key=lambda x: len(x[1]))
            matched_phrase = longest_match[1]
            logger.debug(f"Filtered hallucination: '{matched_phrase}' from '{text}'")
            return True, matched_phrase
        
        return False, ""
    
    def get_stats(self) -> dict:
        """Get statistics about loaded phrases."""
        return {
            'phrases_loaded': self.phrase_count,
            'automaton_ready': self.automaton is not None,
            'csv_path': self.csv_path
        }

class SRTGenerator:
    """Generates SRT subtitle files with proper timestamps."""
    
    def __init__(self, output_file: str = SRT_OUTPUT_FILE):
        self.output_file = output_file
        self.subtitles = []
        self.start_time = None
        self.subtitle_counter = 1
        
    def start_session(self):
        """Start a new subtitle session."""
        self.start_time = time.time()
        self.subtitles = []
        self.subtitle_counter = 1
        logger.info(f"Started SRT subtitle generation - output: {self.output_file}")
    
    def add_subtitle(self, text: str, duration: float = 3.0):
        """Add a subtitle entry with timestamp."""
        if not self.start_time or not text.strip():
            return
            
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # Calculate start and end times for this subtitle
        start_seconds = relative_time - duration
        end_seconds = relative_time
        
        # Ensure start time is not negative
        start_seconds = max(0, start_seconds)
        
        subtitle_entry = {
            'index': self.subtitle_counter,
            'start': start_seconds,
            'end': end_seconds,
            'text': text.strip()
        }
        
        self.subtitles.append(subtitle_entry)
        self.subtitle_counter += 1
        
        logger.debug(f"Added SRT entry #{self.subtitle_counter-1}: {text[:50]}...")
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def save_srt_file(self) -> bool:
        """Save accumulated subtitles to SRT file."""
        if not self.subtitles:
            logger.warning("No subtitles to save")
            return False
            
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for subtitle in self.subtitles:
                    # SRT format:
                    # 1
                    # 00:00:00,000 --> 00:00:03,000
                    # Subtitle text
                    # (blank line)
                    
                    f.write(f"{subtitle['index']}\n")
                    f.write(f"{self.format_timestamp(subtitle['start'])} --> {self.format_timestamp(subtitle['end'])}\n")
                    f.write(f"{subtitle['text']}\n")
                    f.write("\n")  # Blank line between entries
            
            logger.info(f"Saved {len(self.subtitles)} subtitles to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save SRT file: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get subtitle generation statistics."""
        total_duration = 0
        if self.subtitles and self.start_time:
            total_duration = time.time() - self.start_time
            
        return {
            'subtitle_count': len(self.subtitles),
            'total_duration_seconds': total_duration,
            'output_file': self.output_file,
            'session_active': self.start_time is not None
        }

class SmartSubtitleFilter:
    """Filters subtitles to reduce redundancy with intelligent similarity detection."""
    
    def __init__(self, 
                 min_text_length: int = MIN_TEXT_LENGTH,
                 min_word_count: int = MIN_WORD_COUNT,
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 min_display_interval: float = MIN_DISPLAY_INTERVAL,
                 max_repetition_time: float = MAX_REPETITION_TIME):
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count
        self.similarity_threshold = similarity_threshold
        self.min_display_interval = min_display_interval
        self.max_repetition_time = max_repetition_time
        
        # Store recent subtitles for similarity and timing checks
        self.recent_subtitles = []
        self.last_display_time = 0
        
    def should_display(self, text: str) -> tuple[bool, str]:
        """
        Determine if subtitle should be displayed using smart filtering.
        
        Returns:
            (should_display: bool, reason: str)
        """
        text = text.strip()
        current_time = time.time()
        
        # Basic length checks
        if len(text) < self.min_text_length:
            return False, f"Too short ({len(text)} < {self.min_text_length} chars)"
        
        words = text.split()
        if len(words) < self.min_word_count:
            return False, f"Too few words ({len(words)} < {self.min_word_count} words)"
        
        # Time interval check - prevent rapid-fire subtitles
        if current_time - self.last_display_time < self.min_display_interval:
            return False, f"Too soon ({current_time - self.last_display_time:.1f}s < {self.min_display_interval}s)"
        
        # Clean old entries
        self.recent_subtitles = [
            (subtitle_text, timestamp) for subtitle_text, timestamp in self.recent_subtitles
            if current_time - timestamp <= self.max_repetition_time
        ]
        
        # Smart similarity check with recent subtitles
        for recent_text, timestamp in self.recent_subtitles:
            similarity = self._calculate_smart_similarity(text.lower(), recent_text.lower())
            if similarity >= self.similarity_threshold:
                return False, f"Too similar ({similarity:.2f} >= {self.similarity_threshold}) to recent: '{recent_text[:30]}...'"
        
        # Passed all checks - add to recent and update timing
        self.recent_subtitles.append((text, current_time))
        self.last_display_time = current_time
        
        return True, "Passed all filters"
    
    def _calculate_smart_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity with special handling for overlapping speech patterns."""
        if not text1 or not text2:
            return 0.0
        
        # Check for substring containment (common with overlapping chunks)
        if text1 in text2 or text2 in text1:
            longer = text1 if len(text1) > len(text2) else text2
            shorter = text2 if longer == text1 else text1
            return len(shorter) / len(longer)
        
        # Split into words for comparison
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Additional check for overlapping phrases (common with 50% overlap chunks)
        # Check if one text starts where another ends
        words1_list = text1.split()
        words2_list = text2.split()
        
        # Look for overlapping sequences of 3+ words
        for i in range(len(words1_list) - 2):
            phrase1 = ' '.join(words1_list[i:i+3])
            if phrase1 in text2:
                # Found overlapping 3-word phrase, increase similarity
                jaccard_similarity = min(1.0, jaccard_similarity + 0.3)
        
        return jaccard_similarity
    
    def get_stats(self) -> dict:
        """Get filter statistics."""
        return {
            'recent_subtitles_count': len(self.recent_subtitles),
            'min_text_length': self.min_text_length,
            'min_word_count': self.min_word_count,
            'similarity_threshold': self.similarity_threshold,
            'min_display_interval': self.min_display_interval,
            'last_display_time': self.last_display_time
        }

class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD or simple energy-based detection."""
    
    def __init__(self, use_silero=True, energy_threshold=0.01, min_speech_duration=0.5):
        self.use_silero = use_silero and VAD_AVAILABLE
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        self.sample_rate = SAMPLE_RATE
        
        # Silero VAD model
        self.vad_model = None
        self.speech_segments = []
        self.last_speech_time = 0
        
        if self.use_silero:
            self._load_silero_vad()
        
        logger.info(f"VAD initialized: {'Silero VAD' if self.use_silero else 'Energy-based'}")
    
    def _load_silero_vad(self):
        """Load Silero VAD model."""
        try:
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.vad_model = model
            self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}, falling back to energy-based detection")
            self.use_silero = False
    
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as float32 numpy array
            
        Returns:
            True if speech detected, False otherwise
        """
        if self.use_silero and self.vad_model:
            return self._silero_detect_speech(audio_chunk)
        else:
            return self._energy_detect_speech(audio_chunk)
    
    def _silero_detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Use Silero VAD to detect speech."""
        try:
            # Ensure audio is 16kHz for Silero VAD
            if len(audio_chunk) == 0:
                return False
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=int(self.min_speech_duration * 1000),
                min_silence_duration_ms=100,
                return_seconds=True
            )
            
            # Check if any speech was detected
            has_speech = len(speech_timestamps) > 0
            
            if has_speech:
                self.last_speech_time = time.time()
                logger.debug(f"Silero VAD detected {len(speech_timestamps)} speech segments")
            
            return has_speech
            
        except Exception as e:
            logger.debug(f"Silero VAD error: {e}, falling back to energy detection")
            return self._energy_detect_speech(audio_chunk)
    
    def _energy_detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based speech detection."""
        if len(audio_chunk) == 0:
            return False
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        has_speech = rms_energy > self.energy_threshold
        
        if has_speech:
            self.last_speech_time = time.time()
            logger.debug(f"Energy VAD: {rms_energy:.4f} > {self.energy_threshold}")
        
        return has_speech
    
    def is_silence_period(self, silence_duration_threshold=2.0) -> bool:
        """Check if we're in a prolonged silence period."""
        if self.last_speech_time == 0:
            return False
        
        return (time.time() - self.last_speech_time) > silence_duration_threshold
    
    def get_stats(self) -> dict:
        """Get VAD statistics."""
        return {
            'vad_type': 'Silero VAD' if self.use_silero else 'Energy-based',
            'model_loaded': self.vad_model is not None,
            'energy_threshold': self.energy_threshold,
            'min_speech_duration': self.min_speech_duration,
            'last_speech_time': self.last_speech_time
        }

class PerformanceMonitor:
    """Monitors latency and performance across the audio processing pipeline."""
    
    def __init__(self):
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all performance statistics."""
        self.audio_capture_times = []
        self.vad_processing_times = []
        self.transcription_times = []
        self.total_pipeline_times = []
        
        self.chunk_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        
        self.last_audio_time = None
        self.last_transcription_time = None
        self.current_lag = 0.0
        
        # Timing buckets for current chunk
        self.current_timings = {}
        
        logger.info("Performance monitor reset")
    
    def start_timing(self, stage: str):
        """Start timing for a processing stage."""
        self.current_timings[stage + '_start'] = time.time()
    
    def end_timing(self, stage: str):
        """End timing for a processing stage and record duration."""
        start_key = stage + '_start'
        if start_key in self.current_timings:
            duration = time.time() - self.current_timings[start_key]
            
            # Store timing in appropriate bucket
            if stage == 'audio_capture':
                self.audio_capture_times.append(duration)
            elif stage == 'vad_processing':
                self.vad_processing_times.append(duration)
            elif stage == 'transcription':
                self.transcription_times.append(duration)
            elif stage == 'total_pipeline':
                self.total_pipeline_times.append(duration)
            
            # Keep only last 100 measurements
            for timing_list in [self.audio_capture_times, self.vad_processing_times, 
                               self.transcription_times, self.total_pipeline_times]:
                if len(timing_list) > 100:
                    timing_list.pop(0)
            
            return duration
        return 0.0
    
    def update_lag_metrics(self, audio_duration: float):
        """Update lag calculations based on audio chunk processing."""
        current_time = time.time()
        
        if self.last_audio_time is None:
            self.last_audio_time = current_time
            self.last_transcription_time = current_time
            return
        
        # Calculate expected vs actual timing
        expected_interval = audio_duration
        actual_interval = current_time - self.last_transcription_time
        
        # Lag is the difference between actual and expected processing time
        self.current_lag = max(0, actual_interval - expected_interval)
        
        self.last_transcription_time = current_time
        self.processed_count += 1
    
    def increment_chunk_count(self):
        """Increment total chunk counter."""
        self.chunk_count += 1
    
    def increment_skipped_count(self):
        """Increment skipped chunk counter."""
        self.skipped_count += 1
    
    def is_lagging(self, threshold_seconds: float = 3.0) -> bool:
        """Check if system is significantly lagging."""
        return self.current_lag > threshold_seconds
    
    def should_reset_buffers(self, lag_threshold: float = 5.0, processed_threshold: int = 10) -> bool:
        """Determine if buffers should be reset due to excessive lag."""
        return (self.current_lag > lag_threshold and 
                self.processed_count > processed_threshold)
    
    def get_average_timing(self, timing_list: list) -> float:
        """Get average timing from a list."""
        return sum(timing_list) / len(timing_list) if timing_list else 0.0
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report."""
        return {
            'chunk_statistics': {
                'total_chunks': self.chunk_count,
                'processed_chunks': self.processed_count,
                'skipped_chunks': self.skipped_count,
                'processing_rate': (self.processed_count / max(1, self.chunk_count)) * 100
            },
            'timing_averages': {
                'audio_capture_ms': self.get_average_timing(self.audio_capture_times) * 1000,
                'vad_processing_ms': self.get_average_timing(self.vad_processing_times) * 1000,
                'transcription_ms': self.get_average_timing(self.transcription_times) * 1000,
                'total_pipeline_ms': self.get_average_timing(self.total_pipeline_times) * 1000
            },
            'latency_metrics': {
                'current_lag_seconds': self.current_lag,
                'is_lagging': self.is_lagging(),
                'should_reset': self.should_reset_buffers()
            },
            'recent_performance': {
                'last_10_transcription_times': self.transcription_times[-10:] if len(self.transcription_times) >= 10 else self.transcription_times,
                'last_10_total_pipeline_times': self.total_pipeline_times[-10:] if len(self.total_pipeline_times) >= 10 else self.total_pipeline_times
            }
        }
    
    def log_performance_summary(self):
        """Log a summary of current performance metrics."""
        report = self.get_performance_report()
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Chunks: {report['chunk_statistics']['total_chunks']} total, "
                   f"{report['chunk_statistics']['processed_chunks']} processed, "
                   f"{report['chunk_statistics']['skipped_chunks']} skipped "
                   f"({report['chunk_statistics']['processing_rate']:.1f}% rate)")
        
        logger.info(f"Avg Timings: VAD={report['timing_averages']['vad_processing_ms']:.1f}ms, "
                   f"Transcription={report['timing_averages']['transcription_ms']:.1f}ms, "
                   f"Total={report['timing_averages']['total_pipeline_ms']:.1f}ms")
        
        logger.info(f"Latency: Current lag={report['latency_metrics']['current_lag_seconds']:.2f}s, "
                   f"Lagging={report['latency_metrics']['is_lagging']}, "
                   f"Should reset={report['latency_metrics']['should_reset']}")
        
        if report['latency_metrics']['is_lagging']:
            logger.warning("⚠️  System is experiencing significant lag")
        
        if report['latency_metrics']['should_reset']:
            logger.warning("⚠️  Consider resetting audio buffers due to excessive lag")

class SubtitleFilter:
    """Filters subtitles to reduce redundancy and improve readability."""
    
    def __init__(self, 
                 min_text_length: int = None,
                 min_word_count: int = None,
                 similarity_threshold: float = None,
                 max_repetition_time: float = None):
        
        # Set defaults if not provided
        self.min_text_length = min_text_length if min_text_length is not None else MIN_TEXT_LENGTH
        self.min_word_count = min_word_count if min_word_count is not None else MIN_WORD_COUNT
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
        self.max_repetition_time = max_repetition_time if max_repetition_time is not None else MAX_REPETITION_TIME
        
        # Store recent subtitles for similarity checking
        self.recent_subtitles = []
        
    def should_display(self, text: str) -> tuple[bool, str]:
        """
        Determine if subtitle should be displayed.
        
        Returns:
            (should_display: bool, reason: str)
        """
        text = text.strip()
        
        # Check minimum length
        if len(text) < self.min_text_length:
            return False, f"Too short ({len(text)} < {self.min_text_length} chars)"
        
        # Check minimum word count
        words = text.split()
        if len(words) < self.min_word_count:
            return False, f"Too few words ({len(words)} < {self.min_word_count} words)"
        
        # Check similarity with recent subtitles
        current_time = time.time()
        
        # Clean old subtitles
        self.recent_subtitles = [
            (subtitle_text, timestamp) for subtitle_text, timestamp in self.recent_subtitles
            if current_time - timestamp <= self.max_repetition_time
        ]
        
        # Check similarity with recent subtitles
        for recent_text, timestamp in self.recent_subtitles:
            similarity = self._calculate_similarity(text.lower(), recent_text.lower())
            if similarity >= self.similarity_threshold:
                return False, f"Too similar ({similarity:.2f} >= {self.similarity_threshold}) to recent subtitle"
        
        # Add to recent subtitles
        self.recent_subtitles.append((text, current_time))
        
        return True, "Passed all filters"
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word-based approach."""
        if not text1 or not text2:
            return 0.0
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_stats(self) -> dict:
        """Get filter statistics."""
        return {
            'recent_subtitles_count': len(self.recent_subtitles),
            'min_text_length': self.min_text_length,
            'min_word_count': self.min_word_count,
            'similarity_threshold': self.similarity_threshold
        }
    """Detects and filters out hallucinated phrases using Aho-Corasick algorithm."""
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = HALLUCINATION_CSV
        self.csv_path = csv_path
        self.automaton = None
        self.phrase_count = 0
        
    def load_phrases(self) -> bool:
        """Load hallucination phrases from CSV file."""
        if not HALLUCINATION_DETECTION_AVAILABLE:
            logger.warning("Hallucination detection not available, phrases will not be filtered")
            return False
            
        if not os.path.exists(self.csv_path):
            logger.warning(f"Hallucination CSV file not found: {self.csv_path}")
            logger.info("Continuing without hallucination filtering")
            return False
        
        try:
            # Load CSV
            df = pd.read_csv(self.csv_path, encoding="utf-8-sig").dropna()
            df.columns = df.columns.str.strip()
            
            if 'Phrase' not in df.columns:
                logger.error(f"CSV must contain a 'Phrase' column. Found: {df.columns.tolist()}")
                return False
            
            # Process phrases
            phrases = (df['Phrase']
                      .astype(str)
                      .str.strip()
                      .str.lower()
                      .drop_duplicates())
            
            # Remove empty phrases
            phrases = phrases[phrases.str.len() > 0]
            
            if phrases.empty:
                logger.warning(f"No valid phrases found in {self.csv_path}")
                return False
            
            # Build Aho-Corasick automaton
            self.automaton = ahocorasick.Automaton()
            for phrase in phrases:
                self.automaton.add_word(phrase, phrase)
            
            self.automaton.make_automaton()
            self.phrase_count = len(phrases)
            
            logger.info(f"Loaded {self.phrase_count} hallucination phrases for filtering")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hallucination phrases: {e}")
            return False
    
    def is_hallucination(self, text: str) -> tuple[bool, str]:
        """
        Check if text contains hallucinated phrases.
        
        Returns:
            (is_hallucination: bool, matched_phrase: str)
        """
        if not self.automaton:
            return False, ""
        
        text_lower = text.lower().strip()
        
        # Find all matches
        matches = list(self.automaton.iter(text_lower))
        
        if matches:
            # Return the longest matched phrase
            longest_match = max(matches, key=lambda x: len(x[1]))
            matched_phrase = longest_match[1]
            logger.debug(f"Filtered hallucination: '{matched_phrase}' from '{text}'")
            return True, matched_phrase
        
        return False, ""
    
    def get_stats(self) -> dict:
        """Get statistics about loaded phrases."""
        return {
            'phrases_loaded': self.phrase_count,
            'automaton_ready': self.automaton is not None,
            'csv_path': self.csv_path
        }

class DirectFasterWhisperClient:
    def __init__(self):
        self.device_index = None
        self.is_streaming = False
        self.shutdown_requested = False
        self.whisper_model = None
        self.total_bytes_processed = 0
        self.audio_buffer = []
        self.processing_queue = queue.Queue(maxsize=10)
        
        # Web server components
        self.flask_app = None
        self.socketio = None
        self.web_thread = None
        
        # Hallucination detection
        self.hallucination_detector = HallucinationDetector()
        self.filtered_count = 0
        
        # SRT subtitle generation
        self.srt_generator = SRTGenerator()
        
        # Voice Activity Detection
        self.vad = VoiceActivityDetector(use_silero=True)
        self.silence_count = 0
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.last_performance_log = time.time()
        
        # Smart subtitle filtering  
        self.subtitle_filter = SmartSubtitleFilter()
        self.redundant_count = 0
        
    def load_device_index(self):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f).get("input_device_index")
        except Exception as e:
            logger.warning(f"Failed to read config: {e}")
            return None

    def resolve_device(self, device_index):
        """Resolve device index to proper format."""
        if isinstance(device_index, str):
            if device_index.startswith(("hw:", "plughw:")):
                return device_index
            elif device_index.isdigit():
                return int(device_index)
            else:
                raise ValueError(f"Invalid device string: '{device_index}'")
        return device_index

    def setup_web_server(self, host=WEB_HOST, port=WEB_PORT):
        """Setup Flask web server with SocketIO for live subtitles."""
        if not FLASK_AVAILABLE:
            logger.warning("Flask not available, skipping web server setup")
            return False
            
        try:
            self.flask_app = Flask(__name__)
            self.flask_app.config['SECRET_KEY'] = 'whisper_subtitles_secret'
            
            # Disable Flask logging to avoid cluttering output
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
            
            self.socketio = SocketIO(
                self.flask_app, 
                cors_allowed_origins="*",
                logger=False,
                engineio_logger=False
            )
            
            # Define routes
            @self.flask_app.route('/')
            def index():
                return render_template_string(SUBTITLE_HTML)
            
            @self.socketio.on('connect')
            def handle_connect():
                logger.info(f"Web client connected for subtitles")
                emit('subtitle', {'text': 'Connected to live subtitles', 'confidence': 0})
            
            @self.socketio.on('disconnect')
            def handle_disconnect():
                logger.info("Web client disconnected")
            
            logger.info(f"Web server setup complete on http://{host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup web server: {e}")
            return False

    def start_web_server(self, host=WEB_HOST, port=WEB_PORT):
        """Start the web server in a separate thread."""
        if not self.flask_app or not self.socketio:
            return False
            
        def run_server():
            try:
                logger.info(f"Starting subtitle web server on http://{host}:{port}")
                self.socketio.run(
                    self.flask_app,
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False,
                    log_output=False
                )
            except Exception as e:
                logger.error(f"Web server error: {e}")
        
        self.web_thread = threading.Thread(target=run_server, daemon=True)
        self.web_thread.start()
        return True

    def send_subtitle(self, text, confidence=0.0):
        """Send subtitle to connected web clients."""
        if self.socketio:
            try:
                self.socketio.emit('subtitle', {
                    'text': text,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.debug(f"Error sending subtitle: {e}")

    def clear_subtitles(self):
        """Clear all subtitles on web clients."""
        if self.socketio:
            try:
                self.socketio.emit('clear')
            except Exception as e:
                logger.debug(f"Error clearing subtitles: {e}")

    def setup_whisper_model(self, model_size="large-v2", device="cuda", compute_type="float16"):
        """Initialize faster-whisper model directly."""
        try:
            logger.info(f"Initializing faster-whisper model: {model_size}")
            
            # Available device types: "cpu", "cuda", "auto"
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            logger.info(f"Using device: {device}, compute_type: {compute_type}")
            
            self.whisper_model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                # Additional optimizations
                local_files_only=False,  # Allow download if not cached
                num_workers=1  # Single worker for real-time
            )
            
            logger.info("Faster-whisper model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize whisper model: {e}")
            return False

    def convert_float32_to_float32_array(self, audio_data):
        """Convert sounddevice float32 to numpy array suitable for whisper."""
        # Ensure it's a 1D array
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Whisper expects float32 values between -1 and 1
        audio_data = np.clip(audio_data, -1.0, 1.0)
        return audio_data.astype(np.float32)

    def transcribe_audio_chunk(self, audio_chunk):
        """Transcribe a chunk of audio using faster-whisper."""
        try:
            if not self.whisper_model or len(audio_chunk) == 0:
                return None
            
            # Convert to the format whisper expects
            audio_array = self.convert_float32_to_float32_array(audio_chunk)
            
            # Skip very short audio chunks
            duration = len(audio_array) / SAMPLE_RATE
            if duration < MIN_CHUNK_DURATION:
                return None
            
            logger.debug(f"Transcribing {duration:.2f}s audio chunk")
            
            # Transcribe using faster-whisper
            segments, info = self.whisper_model.transcribe(
                audio_array,
                language="en",  # Change as needed
                task="transcribe",
                # Real-time optimizations
                beam_size=1,  # Faster but less accurate
                best_of=1,    # Single candidate
                temperature=0.0,  # Deterministic
                # Silence handling
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                # Chunk handling
                initial_prompt=None,
                condition_on_previous_text=True
            )
            
            # Collect all text from segments
            full_text = ""
            for segment in segments:
                if segment.text.strip():
                    full_text += segment.text.strip() + " "
            
            if full_text.strip():
                detected_language = info.language
                confidence = getattr(info, 'language_probability', 0.0)
                return {
                    "text": full_text.strip(),
                    "language": detected_language,
                    "confidence": confidence,
                    "duration": duration
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing audio chunk: {e}")
            return None

    def audio_processor_worker(self):
        """Worker thread to process audio chunks."""
        logger.info("Audio processor worker started")
        
        while not self.shutdown_requested:
            try:
                # Get audio chunk from queue
                try:
                    audio_chunk = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if audio_chunk is None:  # Shutdown signal
                    break
                # Start performance timing for this chunk
                self.performance_monitor.start_timing('total_pipeline')
                self.performance_monitor.increment_chunk_count()

                # Transcribe the chunk
                result = self.transcribe_audio_chunk(audio_chunk)
                
                # First check VAD - skip if no speech detected
                if not self.vad.detect_speech(audio_chunk):
                    self.silence_count += 1
                    if self.silence_count % 10 == 0:  # Log every 10 silent chunks
                        logger.debug(f"[VAD-SILENCE] No speech detected in chunk #{self.silence_count}")
                    continue
                
                if result:
                    # Check for hallucinations before displaying
                    is_hallucination, matched_phrase = self.hallucination_detector.is_hallucination(result['text'])
                    
                    if is_hallucination:
                        self.filtered_count += 1
                        logger.info(f"[FILTERED-HALLUCINATION] '{matched_phrase}' in '{result['text']}'")
                        # Skip this transcription - don't display or send to web
                        continue
                    
                    # Use SmartSubtitleFilter instead of inline filtering
                    should_display, filter_reason = self.subtitle_filter.should_display(result['text'])

                    if not should_display:
                        self.redundant_count += 1
                        logger.debug(f"[FILTERED-SMART] {filter_reason}: '{result['text'][:50]}...'")
                        continue

                    # Display result in console (only if passed all filters)
                    confidence_info = f" (confidence: {result['confidence']:.2f})" if result['confidence'] > 0 else ""
                    duration_info = f" [{result['duration']:.1f}s]"
                    language_info = f" [{result['language']}]" if result['language'] != 'en' else ""
                    
                    console_output = f"[TRANSCRIPT]{language_info}{duration_info}{confidence_info} {result['text']}"
                    print(console_output)
                    
                    # Send to web interface for subtitles (only if passed all filters)
                    self.send_subtitle(result['text'], result['confidence'])
                    
                    # Add to SRT subtitle file
                    self.srt_generator.add_subtitle(result['text'], result['duration'])

                # Update performance metrics
                if result:
                    self.performance_monitor.update_lag_metrics(result['duration'])
                else:
                    self.performance_monitor.increment_skipped_count()

                # End performance timing
                self.performance_monitor.end_timing('total_pipeline')

                # Log performance summary periodically
                current_time = time.time()
                if current_time - self.last_performance_log > 30:  # Every 30 seconds
                    self.performance_monitor.log_performance_summary()
                    self.last_performance_log = current_time

                # Check if system is lagging and should reset buffers
                if self.performance_monitor.should_reset_buffers():
                    logger.warning("Resetting audio buffer due to excessive lag")
                    self.audio_buffer.clear()
                    # Clear processing queue
                    while not self.processing_queue.empty():
                        try:
                            self.processing_queue.get_nowait()
                        except queue.Empty:
                            break

                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audio processor worker: {e}")
        
        logger.info("Audio processor worker stopped")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input from sounddevice."""
        if status:
            if "overflow" in str(status):
                logger.warning(f"Audio overflow: {status}")

        if self.shutdown_requested:
            return

        try:
            # Add audio to buffer
            audio_data = indata[:, 0]  # Get mono channel
            self.audio_buffer.extend(audio_data)
            
            # Check if we have enough audio for processing
            samples_needed = int(SAMPLE_RATE * CHUNK_DURATION)
            
            if len(self.audio_buffer) >= samples_needed:
                # Extract chunk for processing
                chunk = np.array(self.audio_buffer[:samples_needed])
                self.audio_buffer = self.audio_buffer[samples_needed//2:]  # 50% overlap
                
                # Add to processing queue
                try:
                    self.processing_queue.put(chunk, timeout=0.01)
                except queue.Full:
                    # Drop oldest chunk if queue is full
                    try:
                        self.processing_queue.get_nowait()
                        self.processing_queue.put(chunk, timeout=0.01)
                    except (queue.Empty, queue.Full):
                        pass
                    
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    async def stream_audio(self, device_index):
        """Start audio streaming from USB microphone."""
        logger.info(f"Starting audio stream with device {device_index}")

        device_arg = self.resolve_device(device_index)
        
        # Start the audio processor worker thread
        processor_thread = threading.Thread(target=self.audio_processor_worker, daemon=True)
        processor_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=BUFFER_SIZE,
                callback=self.audio_callback,
                device=device_arg
            ):
                logger.info("Audio stream started, processing with faster-whisper")
                logger.info(f"Processing audio in {CHUNK_DURATION}s chunks with 50% overlap")
                self.is_streaming = True
                
                # Keep running until shutdown
                while not self.shutdown_requested:
                    await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            raise
        finally:
            self.is_streaming = False
            
            # Signal worker thread to stop
            self.processing_queue.put(None)
            processor_thread.join(timeout=5.0)
            
            logger.info("Audio stream stopped")

    async def run(self, model_size="large-v2", device="cuda", compute_type="float16", enable_web=True, enable_hallucination_filter=True, enable_srt=True):
        """Main run method."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper not available")
            return False
            
        logger.info("Starting Direct Faster-Whisper USB Mic Client")
        
        # Load device configuration
        device_index = self.load_device_index()
        if device_index is None:
            logger.error("No input device index found in config file")
            logger.info("Create a usb_mic_config.json file with: {\"input_device_index\": YOUR_DEVICE_INDEX}")
            return False

        # Setup Whisper model
        if not self.setup_whisper_model(model_size=model_size, device=device, compute_type=compute_type):
            return False

        # Setup hallucination detection
        if enable_hallucination_filter:
            if self.hallucination_detector.load_phrases():
                stats = self.hallucination_detector.get_stats()
                logger.info(f"Hallucination filtering enabled with {stats['phrases_loaded']} phrases")
            else:
                logger.warning("Hallucination filtering disabled due to loading issues")

        # Start SRT subtitle generation
        if enable_srt:
            self.srt_generator.start_session()

        # Setup and start web server if enabled
        if enable_web:
            if self.setup_web_server():
                if self.start_web_server():
                    logger.info(f"Subtitle interface available at: http://localhost:{WEB_PORT}")
                    logger.info("Use this URL as a Browser Source in OBS Studio")
                else:
                    logger.warning("Failed to start web server, continuing without web interface")
            else:
                logger.warning("Failed to setup web server, continuing without web interface")

        try:
            # Start audio streaming
            await self.stream_audio(device_index)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
            
        finally:
            if enable_web:
                self.clear_subtitles()
            
            # Save SRT file before shutdown
            if enable_srt:
                srt_stats = self.srt_generator.get_stats()
                if self.srt_generator.save_srt_file():
                    logger.info(f"SRT file saved with {srt_stats['subtitle_count']} entries, "
                               f"duration: {srt_stats['total_duration_seconds']:.1f}s")
            
            # Display final statistics
            if enable_hallucination_filter and self.filtered_count > 0:
                logger.info(f"Filtered {self.filtered_count} hallucinated phrases during session")
            
            if self.silence_count > 0:
                logger.info(f"Skipped {self.silence_count} silent audio chunks (VAD filtering)")
            
            if self.redundant_count > 0:
                logger.info(f"Filtered {self.redundant_count} redundant/duplicate subtitles")
            
            # Show subtitle filter settings
            subtitle_stats = self.subtitle_filter.get_stats()
            logger.info(f"Smart filter: {subtitle_stats['recent_subtitles_count']} recent subtitles tracked, "
                        f"{subtitle_stats['similarity_threshold']:.1f} similarity threshold")
            vad_stats = self.vad.get_stats()
            logger.info(f"VAD: {vad_stats['vad_type']}, Energy threshold: {vad_stats['energy_threshold']}")
            # Show final performance report
            logger.info("=== FINAL PERFORMANCE REPORT ===")
            self.performance_monitor.log_performance_summary()
            logger.info("Direct faster-whisper client shutdown complete")

        return True

    def shutdown(self):
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self.shutdown_requested = True

async def main():
    """Main entry point with signal handling."""
    client = DirectFasterWhisperClient()
    
    # Setup signal handlers
    def signal_handler():
        client.shutdown()
    
    if sys.platform != "win32":
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)
        except Exception as e:
            logger.debug(f"Could not setup signal handlers: {e}")
    
    # Configuration options
    MODEL_SIZE = "large-v2"     # Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3
    DEVICE = "cuda"             # Options: "cpu", "cuda", "auto"
    COMPUTE_TYPE = "float16"       # Options: "int8", "int16", "float16", "float32"
    ENABLE_WEB = True           # Enable web interface for subtitles
    ENABLE_HALLUCINATION_FILTER = True  # Enable filtering of hallucinated phrases
    ENABLE_SRT = True           # Enable SRT subtitle file generation
    
    try:
        success = await client.run(
            model_size=MODEL_SIZE,
            device=DEVICE, 
            compute_type=COMPUTE_TYPE,
            enable_web=ENABLE_WEB,
            enable_hallucination_filter=ENABLE_HALLUCINATION_FILTER,
            enable_srt=ENABLE_SRT
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in main")
        client.shutdown()

if __name__ == "__main__":
    print("Real-time Speech to Subtitle System with Smart Filtering")
    print("=" * 70)
    print("Features: USB Mic → faster-whisper → Multi-layer Filtering → Clean Subtitles")
    print("")
    print("Filtering Layers:")
    print(f"1. Length Filter: Minimum {MIN_TEXT_LENGTH} characters, {MIN_WORD_COUNT} words")
    print(f"2. Similarity Filter: Skip if {SIMILARITY_THRESHOLD*100:.0f}% similar to recent subtitles")
    print(f"3. Hallucination Filter: Block promotional phrases from CSV")
    print("")
    print("Requirements:")
    print("- pip install faster-whisper sounddevice numpy flask flask-socketio")
    print("- pip install pyahocorasick pandas  # For hallucination filtering")
    print("- USB microphone configured")
    print("")
    print("Setup:")
    print("1. Create usb_mic_config.json with your USB mic device index")
    print("2. Create hallucination_phrases.csv with phrases to filter out:")
    print("   Phrase")
    print("   thank you for watching")
    print("   please like and subscribe")
    print("   terima kasih kerana menonton")
    print("3. Adjust filter settings in script constants if needed:")
    print(f"   MIN_TEXT_LENGTH = {MIN_TEXT_LENGTH}  # Minimum characters")
    print(f"   MIN_WORD_COUNT = {MIN_WORD_COUNT}   # Minimum words")
    print(f"   SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}  # Similarity cutoff (0.0-1.0)")
    print("4. For OBS: Add Browser Source pointing to http://localhost:5000")
    print("")
    print("Web Interface: http://localhost:5000 (for OBS Browser Source)")
    print("=" * 70)
    
    if not FASTER_WHISPER_AVAILABLE:
        print("\nPlease install faster-whisper first:")
        print("pip install faster-whisper")
        sys.exit(1)
        
    if not FLASK_AVAILABLE:
        print("\nPlease install Flask dependencies for web subtitles:")
        print("pip install flask flask-socketio")
        print("(Web interface will be disabled without these)")
    
    if not HALLUCINATION_DETECTION_AVAILABLE:
        print("\nOptional: Install hallucination detection dependencies:")
        print("pip install pyahocorasick pandas")
        print("(Hallucination filtering will be disabled without these)")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
