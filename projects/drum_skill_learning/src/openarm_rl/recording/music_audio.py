import subprocess
import wave

import imageio_ffmpeg
import numpy as np
from scipy.signal import butter, sosfilt


RATE = 48000
# (fundamental Hz, pitch-sweep Hz, tone decay s, noise decay s, noise gain, noise high-pass Hz)
VOICES = {"kick": (55, 70, .09, .006, .3, 1500), "snare": (190, 60, .05, .07, .9, 1200),
          "floor_tom": (85, 40, .14, .01, .15, 1200), "rack_tom_left": (150, 50, .10, .01, .15, 1200),
          "rack_tom_right": (120, 45, .12, .01, .15, 1200), "hi_hat": (0, 0, .01, .03, .5, 7000),
          "cymbal_left": (0, 0, .01, .34, .42, 3500), "cymbal_right": (0, 0, .01, .20, .36, 4500)}


def drum_voice(target, index):
    fundamental, sweep, tone_decay, noise_decay, noise_gain, cutoff = VOICES[target]
    # Six time constants leave the envelope below 0.3 percent, so the sample never ends in a click.
    t = np.arange(round(RATE*6*max(tone_decay, noise_decay)))/RATE
    noise = sosfilt(butter(3, cutoff, btype="highpass", fs=RATE, output="sos"), np.random.default_rng(index).normal(size=len(t)))
    pulse = noise_gain*noise*np.exp(-t/noise_decay)
    if fundamental:
        phase = fundamental*t+sweep*.03*(1-np.exp(-t/.03))
        pulse = pulse+np.sin(2*np.pi*phase)*np.exp(-t/tone_decay)
    return pulse


def mix_contacts(hits, preroll, duration, backing=()):
    samples = np.zeros(max(1, round(duration*RATE)))
    for index, hit in enumerate((*hits, *backing)):
        pulse = drum_voice(hit["target"], index)
        if "midi_velocity" in hit:
            pulse = pulse*float(hit["midi_velocity"])/127
        offset = round((hit["time_s"]+preroll)*RATE)
        end = min(len(samples), offset+len(pulse))
        if offset >= 0 and end > offset:
            samples[offset:end] += pulse[:end-offset]
    return samples


def add_contact_audio(directory, hits, preroll, duration, backing=()):
    samples = mix_contacts(hits, preroll, duration, backing)
    peak = max(1., float(np.max(np.abs(samples))))
    pcm = (samples/peak*.8*32767).astype("<i2")
    audio = directory / "contacts.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(directory / "simulation_silent.mp4"),
                    "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(directory / "performance.mp4")], check=True)
