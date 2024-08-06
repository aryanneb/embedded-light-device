import time
import board
import neopixel_spi as neopixel
import numpy as np
import pyaudio
import librosa
from collections import deque

# Configuration
NUM_PIXELS = 21
PIXEL_ORDER = neopixel.GRB
DELAY = 0.01  # small delay for responsiveness
SENSITIVITY = 4.0  # sensitivity multiplier for brightness
BRIGHTNESS_SCALE = 1.5  # scaling for brightness increase with more LEDs lit
DECAY_RATE = 7.0  # decay rate for brightness drop
SMOOTHING_WINDOW_SIZE = 10  # window size for smoothing decibel changes
COLOR_SMOOTHING = 0.1  # smoothing factor for color transitions (0-1)

# frequency bands (in hz)
LOW_FREQ_BAND = (20, 250)    # low frequencies (bass)
MID_FREQ_BAND = (250, 2000)  # mid frequencies
HIGH_FREQ_BAND = (2000, 8000)  # high frequencies (treble)

# setting up neopixel spi
spi = board.SPI()
pixels = neopixel.NeoPixel_SPI(spi, NUM_PIXELS, pixel_order=PIXEL_ORDER, auto_write=False)

# audio settings
CHUNK = 1024  # number of audio samples per frame
FORMAT = pyaudio.paInt16  # format for pyaudio
CHANNELS = 1  # mono audio
RATE = 44100  # sample rate

# initializing pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

current_brightness = 0  # set def brightness
previous_decibels = 0  # set prev def decibels

# smoothing decibel readings and colors with deque for EMA
decibel_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
current_color = (0, 0, 0)  # set def color

def get_volume_and_frequencies(data):
    """processes audio data with rms and stft, calculates frequency bands and weighted mean frequency"""
    data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    
    # compute RMS for overall loudness
    rms = np.sqrt(np.mean(np.square(data)))
    
    # compute STFT for frequency analysis
    stft = np.abs(librosa.stft(data, n_fft=1024, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=RATE, n_fft=1024)
    
    # sum energies in each frequency band
    low_energy = np.sum(stft[(frequencies >= LOW_FREQ_BAND[0]) & (frequencies < LOW_FREQ_BAND[1])])
    mid_energy = np.sum(stft[(frequencies >= MID_FREQ_BAND[0]) & (frequencies < MID_FREQ_BAND[1])])
    high_energy = np.sum(stft[(frequencies >= HIGH_FREQ_BAND[0]) & (frequencies < HIGH_FREQ_BAND[1])])
    
    # compute weighted mean frequency
    mean_frequencies = np.sum(stft * frequencies[:, None], axis=0) / np.sum(stft, axis=0)
    mean_freq = np.mean(mean_frequencies)
    
    if rms > 0:
        # convert RMS to decibels
        decibels = 20 * np.log10(rms)
    else:
        decibels = 0  # return 0 db to indicate silence (only really recorded when microphone is off, for testing purposes)
    
    return decibels, low_energy, mid_energy, high_energy, mean_freq

def get_gradient_color(low_energy, mid_energy, high_energy):
    """creates a gradient color based on frequency distribution, leveraging fft and proportional energy analysis"""
    # normalize energies to a 0-1 scale
    total_energy = low_energy + mid_energy + high_energy
    if total_energy == 0:
        return (0, 0, 0)
    
    # increase sensitivity by amplifying the ratios
    low_ratio = (low_energy / total_energy) ** 2
    mid_ratio = (mid_energy / total_energy) ** 2
    high_ratio = (high_energy / total_energy) ** 2

    # color blending: purple (low) -> blue (mid) -> teal (high)
    red = int(255 * low_ratio)
    green = int(255 * high_ratio)
    blue = int(128 * low_ratio + 255 * mid_ratio + 255 * high_ratio)

    return (red, green, blue)

def smooth_color(new_color, old_color, smoothing_factor):
    """smoothly transitions between colors using exponential moving average"""
    return tuple(int(old_c + smoothing_factor * (new_c - old_c)) for new_c, old_c in zip(new_color, old_color))

def set_pixels_brightness_and_color(decibels, color):
    """adjusts brightness and color based on decibel levels, with a center-outward color spread"""
    mid_point = NUM_PIXELS // 2
    
    for i in range(mid_point + 1):
        sensitivity_scale = 1.0 - (i / mid_point)  # center is most sensitive, outer edges are less
        threshold = 40 + (i * 5)  # lower thresholds for more sensitivity
        
        if decibels >= threshold:
            brightness = int((decibels - threshold) * SENSITIVITY * sensitivity_scale * BRIGHTNESS_SCALE)
            brightness = max(min(brightness, 255), 0)
        else:
            brightness = 0  # led remains off if below threshold
        
        scaled_color = tuple(int(c * (brightness / 255)) for c in color)
        
        # set leds symmetrically from the center
        if mid_point + i < NUM_PIXELS:
            pixels[mid_point + i] = scaled_color
        if mid_point - i >= 0:
            pixels[mid_point - i] = scaled_color
    
    pixels.show()

# start processing real-time audio input with continuous updates
while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    
    # calculate volume and frequency bands
    decibels, low_energy, mid_energy, high_energy, mean_freq = get_volume_and_frequencies(data)
    
    # get gradient color based on frequency bands
    new_color = get_gradient_color(low_energy, mid_energy, high_energy)
    
    # smooth the color transitions with ema
    current_color = smooth_color(new_color, current_color, COLOR_SMOOTHING)
    
    # set pixels brightness and color based on the current decibels and frequencies
    set_pixels_brightness_and_color(np.mean(decibel_history), current_color)
    
    time.sleep(DELAY)  # short delay for responsiveness

# cleanup, but not really necessary as the loop runs indefinitely
stream.stop_stream()
stream.close()
p.terminate()
