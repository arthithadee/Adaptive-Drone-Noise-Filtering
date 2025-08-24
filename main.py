import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

# Constants
TARGET_FS = 48000
DURATION = 20  # seconds

def resample_if_needed(signal, original_fs, target_fs):
    if original_fs != target_fs:
        signal = resample_poly(signal, target_fs, original_fs)
    return signal

def nlms_filter_extract(input_signal, reference, filter_order=128, mu=0.1):
    n_samples = len(input_signal)
    w = np.zeros(filter_order)
    output = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for n in range(filter_order, n_samples):
        x = reference[n-filter_order:n][::-1]
        d = input_signal[n]
        y = np.dot(w, x)
        e = d - y
        norm = np.dot(x, x) + 1e-6
        w += (mu / norm) * e * x

        output[n] = y
        error[n] = e

    return output, error

def plot_signals(input_signal, reference, extracted_output, fs):
    min_len = min(len(input_signal), len(reference), len(extracted_output))
    input_signal = input_signal[:min_len]
    reference = reference[:min_len]
    extracted_output = extracted_output[:min_len]
    t = np.arange(min_len) / fs

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, input_signal)
    plt.title("Input Signal (Microphone Recording)")
    plt.xlabel("Time [s]")

    plt.subplot(3, 1, 2)
    plt.plot(t, reference)
    plt.title("Reference Signal (Drone Template)")
    plt.xlabel("Time [s]")

    plt.subplot(3, 1, 3)
    plt.plot(t, extracted_output)
    plt.title("Extracted Output (Matched via NLMS)")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

def record_from_microphone(duration, fs):
    print(f"[INFO] Recording {duration} seconds from microphone at {fs} Hz...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def process_microphone_input(reference_path, output_path):
    # Record live input
    input_signal = record_from_microphone(DURATION, TARGET_FS)

    # Save raw microphone input
    sf.write("recorded_microphone_input.wav", input_signal, TARGET_FS)
    print(f"[INFO] Raw microphone input saved to: recorded_microphone_input.wav")

    # Load reference
    reference, fs_ref = sf.read(reference_path)
    if reference.ndim > 1:
        reference = reference[:, 0]

    reference = resample_if_needed(reference, fs_ref, TARGET_FS)

    # Repeat (tile) reference to match input length
    if len(reference) < len(input_signal):
        repeat_times = int(np.ceil(len(input_signal) / len(reference)))
        reference = np.tile(reference, repeat_times)[:len(input_signal)]

    print(f"[INFO] Running NLMS drone-extractor...")
    output, residual = nlms_filter_extract(input_signal, reference)

    sf.write(output_path, output, TARGET_FS)
    print(f"[INFO] Extracted drone-like audio saved to: {output_path}")

    plot_signals(input_signal, reference, output, TARGET_FS)


if __name__ == "__main__":
    process_microphone_input(
        reference_path="drone_template_20.wav",
        output_path="filtered_output_nlms_from_mic.wav"
    )
