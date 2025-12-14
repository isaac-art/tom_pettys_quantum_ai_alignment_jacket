import threading
import numpy as np
from pydub import AudioSegment
import pyaudio
import os
from typing import Dict


class AudioController:
    """Controls playback of three audio tracks with speed control based on x/y vectors"""
    
    def __init__(self, audio_dir: str = "audio"):
        self.audio_dir = audio_dir
        self.tracks: Dict[str, np.ndarray] = {}
        self.track_info: Dict[str, dict] = {}
        self.playing = False
        self.current_speeds = {"bass": 1.0, "drums": 1.0, "other": 1.0}
        self.target_speeds = {"bass": 1.0, "drums": 1.0, "other": 1.0}
        self.lock = threading.Lock()
        
        # PyAudio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        self.streams: Dict[str, pyaudio.Stream] = {}
        self.audio_threads: Dict[str, threading.Thread] = {}
        self.playback_positions: Dict[str, float] = {}
        
        # Crossfade parameters for smooth looping
        self.crossfade_samples = 512  # Number of samples for crossfade
        
        # Load audio files
        self._load_tracks()
    
    def _load_tracks(self):
        """Load all three audio tracks and convert to numpy arrays"""
        track_files = {
            "bass": "bass.mp3",
            "drums": "drums.mp3",
            "other": "other.mp3"
        }
        
        for track_name, filename in track_files.items():
            filepath = os.path.join(self.audio_dir, filename)
            if os.path.exists(filepath):
                try:
                    # Load audio file
                    audio = AudioSegment.from_mp3(filepath)
                    
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples())
                    
                    # Handle stereo vs mono
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2)).astype(np.float32)
                        # Convert to mono for simplicity (average channels)
                        samples = samples.mean(axis=1)
                    else:
                        samples = samples.astype(np.float32)
                    
                    # Normalize to [-1, 1]
                    if samples.max() > 0:
                        samples = samples / (2 ** (audio.sample_width * 8 - 1))
                    
                    self.tracks[track_name] = samples
                    self.track_info[track_name] = {
                        "sample_rate": audio.frame_rate,
                        "channels": audio.channels,
                        "sample_width": audio.sample_width
                    }
                    self.playback_positions[track_name] = 0.0
                    
                    print(f"✅ Loaded {track_name} track: {len(samples)} samples @ {audio.frame_rate}Hz")
                except Exception as e:
                    print(f"❌ Failed to load {track_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Audio file not found: {filepath}")
    
    def _calculate_speed(self, x: float, y: float) -> float:
        """Calculate playback speed from x/y vectors (range: 0.25x to 3x)"""
        # Map x from -1 to 1 range to speed multiplier 0.25x to 3x
        # Linear mapping: -1 -> 0.25, 0 -> 1.0, 1 -> 3.0
        base_speed = 0.25 + (x + 1) * 1.375  # Maps -1->0.25, 0->1.625, 1->3.0
        
        # Add y influence (modulation)
        y_mod = y * 0.5  # y: -1 to 1 -> -0.5 to 0.5 speed adjustment
        speed = base_speed + y_mod
        
        # Clamp to valid range
        speed = max(0.25, min(3.0, speed))
        return speed
    
    def update_speeds(self, x: float, y: float):
        """Update playback speeds based on x/y vectors"""
        with self.lock:
            # All tracks use the same speed calculation
            speed = self._calculate_speed(x, y)
            self.target_speeds = {
                "bass": speed,
                "drums": speed,
                "other": speed
            }
    
    def _play_track_loop(self, track_name: str):
        """Play a single track in a loop with real-time speed control"""
        if track_name not in self.tracks:
            return
        
        audio_data = self.tracks[track_name]
        info = self.track_info[track_name]
        sample_rate = info["sample_rate"]
        channels = info["channels"]
        sample_width = info["sample_width"]
        
        chunk_size = 2048  # Larger buffer to reduce underruns
        
        def audio_callback(in_data, frame_count, time_info, status):
            if not self.playing:
                return (None, pyaudio.paComplete)
            
            with self.lock:
                # Smooth speed transitions to avoid clicks
                target_speed = self.target_speeds[track_name]
                current_speed = self.current_speeds[track_name]
                # Smooth interpolation (0.3 smoothing factor for responsive but smooth changes)
                speed = current_speed + (target_speed - current_speed) * 0.3
                self.current_speeds[track_name] = speed
                position = self.playback_positions[track_name]
            
            # Extract samples from current position
            output = np.zeros(frame_count, dtype=np.float32)
            audio_len = len(audio_data)
            
            for i in range(frame_count):
                # Handle wrapping
                while position >= audio_len:
                    position -= audio_len
                while position < 0:
                    position += audio_len
                
                # Get integer position and fractional part
                pos_int = int(position)
                frac = position - pos_int
                
                # Get next position for interpolation
                pos_next = pos_int + 1
                if pos_next >= audio_len:
                    pos_next = 0  # Wrap around
                
                # Linear interpolation for smooth playback
                sample = audio_data[pos_int] * (1 - frac) + audio_data[pos_next] * frac
                
                # Gentle fade near loop boundaries to prevent clicks
                # Only apply very subtle fade to avoid volume loss
                distance_to_start = position
                distance_to_end = audio_len - position
                min_distance = min(distance_to_start, distance_to_end)
                
                if min_distance < self.crossfade_samples:
                    # Very subtle fade (only reduces volume by max 10% at boundary)
                    fade_factor = 0.9 + 0.1 * (min_distance / self.crossfade_samples)
                    sample *= fade_factor
                
                output[i] = sample
                
                # Advance position by speed
                position += speed
            
            with self.lock:
                # Keep position in valid range
                while position >= audio_len:
                    position -= audio_len
                while position < 0:
                    position += audio_len
                self.playback_positions[track_name] = position
            
            # Convert to int16 and then to bytes
            output = np.clip(output, -1.0, 1.0)
            output_int16 = (output * 32767).astype(np.int16)
            
            return (output_int16.tobytes(), pyaudio.paContinue)
        
        try:
            stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,  # We converted to mono
                rate=sample_rate,
                output=True,
                stream_callback=audio_callback,
                frames_per_buffer=chunk_size
            )
            
            self.streams[track_name] = stream
            stream.start_stream()
            
            # Keep thread alive while playing
            while self.playing:
                if stream.is_stopped():
                    break
                threading.Event().wait(0.1)
            
            stream.stop_stream()
            stream.close()
            if track_name in self.streams:
                del self.streams[track_name]
                
        except Exception as e:
            print(f"Error playing {track_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start playing all tracks in loops"""
        if self.playing:
            return
        
        self.playing = True
        
        # Reset playback positions
        for track_name in self.tracks:
            self.playback_positions[track_name] = 0.0
        
        # Start a thread for each track
        for track_name in self.tracks:
            thread = threading.Thread(
                target=self._play_track_loop,
                args=(track_name,),
                daemon=True
            )
            thread.start()
            self.audio_threads[track_name] = thread
        
        print("🎵 Audio playback started")
    
    def stop(self):
        """Stop all audio playback"""
        self.playing = False
        
        # Stop all streams
        for stream in self.streams.values():
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        self.streams.clear()
        
        # Wait for threads to finish
        for thread in self.audio_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.audio_threads.clear()
        
        print("🛑 Audio playback stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

