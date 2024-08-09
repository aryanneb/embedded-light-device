import time
import board
import neopixel_spi as neopixel
import numpy as np
import pyaudio
import librosa
from collections import deque

# setup vibes
NUM_PIXELS = 21
PIXEL_ORDER = neopixel.GRB
DELAY = 0.01
SENSITIVITY = 4.0
BRIGHTNESS_SCALE = 1.5
DECAY_RATE = 7.0
SMOOTHING_WINDOW_SIZE = 10
COLOR_SMOOTHING = 0.25

# frequency bands (where the bass, mids, and treble chill)
LOW_FREQ_BAND = (100, 400)
MID_FREQ_BAND = (400, 700)
HIGH_FREQ_BAND = (700, 1000)

# neopixel setup
spi = board.SPI()
pixels = neopixel.NeoPixel_SPI(spi, NUM_PIXELS, pixel_order=PIXEL_ORDER, auto_write=False)

# audio config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# start audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# smooth out decibel readings and color changes
decibel_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
current_color = (0, 0, 0)

def get_volume_and_frequencies(data):
    """get volume and freq details from the audio"""
    data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    
    # rms for loudness check
    rms = np.sqrt(np.mean(np.square(data)))
    
    # stft for frequency analysis
    stft = np.abs(librosa.stft(data, n_fft=1024, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=RATE, n_fft=1024)
    
    # sum up the energies in each freq band
    low_energy = np.sum(stft[(frequencies >= LOW_FREQ_BAND[0]) & (frequencies < LOW_FREQ_BAND[1])])
    mid_energy = np.sum(stft[(frequencies >= MID_FREQ_BAND[0]) & (frequencies < MID_FREQ_BAND[1])])
    high_energy = np.sum(stft[(frequencies >= HIGH_FREQ_BAND[0]) & (frequencies < HIGH_FREQ_BAND[1])])
    
    # convert rms to decibels, but keep it chill if it's silent
    if rms > 0:
        decibels = 20 * np.log10(rms)
    else:
        decibels = 0
    
    return decibels, low_energy, mid_energy, high_energy

def get_rgb_color(low_energy, mid_energy, high_energy):
    """map those frequency bands to rgb colors"""
    total_energy = low_energy + mid_energy + high_energy
    if total_energy == 0:
        return (0, 0, 0)
    
    # normalize those energies
    low_ratio = (low_energy / total_energy) ** 2
    mid_ratio = (mid_energy / total_energy) ** 2
    high_ratio = (high_energy / total_energy) ** 2

    # assign colors: red for low, green for mid, blue for high
    red = int(255 * low_ratio)
    green = int(255 * mid_ratio)
    blue = int(255 * high_ratio)

    return (red, green, blue)

def smooth_color(new_color, old_color, smoothing_factor):
    """smooth out color transitions like a pro"""
    return tuple(int(old_c + smoothing_factor * (new_c - old_c)) for new_c, old_c in zip(new_color, old_color))

def set_pixels_brightness_and_color(decibels, color):
    """adjust brightness and color based on the loudness, spreading out from the center"""
    mid_point = NUM_PIXELS // 2
    
    for i in range(mid_point + 1):
        sensitivity_scale = 1.0 - (i / mid_point)
        threshold = 40 + (i * 5)
        
        if decibels >= threshold:
            brightness = int((decibels - threshold) * SENSITIVITY * sensitivity_scale * BRIGHTNESS_SCALE)
            brightness = max(min(brightness, 255), 0)
        else:
            brightness = 0
        
        scaled_color = tuple(int(c * (brightness / 255)) for c in color)
        
        # set the leds, mirroring from the center
        if mid_point + i < NUM_PIXELS:
            pixels[mid_point + i] = scaled_color
        if mid_point - i >= 0:
            pixels[mid_point - i] = scaled_color
    
    pixels.show()

# main loop: listen to the audio and update the lights in real-time
while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    
    decibels, low_energy, mid_energy, high_energy = get_volume_and_frequencies(data)
    new_color = get_rgb_color(low_energy, mid_energy, high_energy)
    current_color = smooth_color(new_color, current_color, COLOR_SMOOTHING)
    decibel_history.append(decibels)
    smoothed_decibels = np.mean(decibel_history)
    set_pixels_brightness_and_color(smoothed_decibels, current_color)
    
    time.sleep(DELAY)

# cleanup (but let's be real, this loop is infinite)
stream.stop_stream()
stream.close()
p.terminate()
