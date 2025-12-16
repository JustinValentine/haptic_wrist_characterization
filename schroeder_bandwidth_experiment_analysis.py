import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- CONFIGURATION ---
GEAR_RATIO = 10.0

# Load Data
df = pd.read_csv('bandwidth_data.csv')

# 1. Convert Actual Motor Position to Relative Joint Degrees
# We subtract the initial offset to compare directly with the 0-centered command
start_offset = df['motor_pos_rev'].iloc[0]
df['joint_pos_deg'] = (df['motor_pos_rev'] - start_offset) * (1/GEAR_RATIO) * 360.0

# 2. Prepare Signals
# Input: Command (u)
# Output: Actual (y)
t = df['time'].values
u = df['cmd_pos_deg'].values
y = df['joint_pos_deg'].values

# Calculate Sampling Frequency (fs)
fs = 1.0 / np.mean(np.diff(t))
print(f"Detected Sampling Freq: {fs:.2f} Hz")

# 3. Compute Frequency Response (Transfer Function Estimate)
# We use Welch's method or simply the ratio of FFTs if noise is low. 
# `scipy.signal.csd` (Cross Spectral Density) is robust.
# H(f) = Pxy(f) / Pxx(f)

freqs, Pxx = signal.welch(u, fs, nperseg=1024)
freqs, Pyy = signal.welch(y, fs, nperseg=1024)
freqs, Pxy = signal.csd(u, y, fs, nperseg=1024)

# Transfer Function H
H = Pxy / Pxx
magnitude = np.abs(H)
magnitude_db = 20 * np.log10(magnitude)
phase_deg = np.angle(H, deg=True)

# 4. Find Bandwidth (-3dB point)
# We scan for the first frequency where gain drops below -3dB
bandwidth_hz = np.nan
for f, mag in zip(freqs, magnitude_db):
    if f > 0.5 and mag < -3.0: # Ignore DC/very low freq noise
        bandwidth_hz = f
        break

print("-" * 30)
print("BANDWIDTH RESULTS")
print("-" * 30)
print(f"Closed-Loop Bandwidth (-3dB): {bandwidth_hz:.2f} Hz")
print("-" * 30)

# 5. Plotting (Replicating Fig 3.6 and 3.7) [cite: 959, 978]

# Time Domain Plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, u, 'r-', label='Desired (Cmd)', alpha=0.7)
plt.plot(t, y, 'b--', label='Actual', alpha=0.7)
plt.title("Time Domain Response (Schroeder Multisine)")
plt.ylabel("Position [deg]")
plt.legend()
plt.grid(True)

# Frequency Domain (Bode) Plot
plt.subplot(2, 1, 2)
plt.semilogx(freqs, magnitude_db, 'k-', linewidth=2)
plt.axhline(-3.0, color='r', linestyle='--', label='-3dB Cutoff')
if not np.isnan(bandwidth_hz):
    plt.axvline(bandwidth_hz, color='r', linestyle='--')
    plt.text(bandwidth_hz, -10, f"  {bandwidth_hz:.2f} Hz", color='r')

plt.title("Bode Plot (Closed-Loop Position)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.xlim(0.1, 20) # Match the sweep range
plt.ylim(-20, 5)  # Typical range
plt.grid(True, which="both", ls="-")
plt.legend()

plt.tight_layout()
plt.show()