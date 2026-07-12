"""
Unified Power Quality Disturbance Generator
Covers all unique disturbance categories identified across 18 research papers.
Base: 50 Hz sinusoidal system, IEEE 1159-aligned parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ── System Configuration ──────────────────────────────────────────────────────
FS       = 10000          # 10 kHz (Papers 3, 5, 11)
F0       = 50.0           # 50 Hz fundamental
OMEGA    = 2 * np.pi * F0
DURATION = 0.4            # 400 ms
T        = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
PER      = 1.0 / F0       # one cycle = 20 ms

# Disturbance window: cycles 3-7 out of 20-cycle window
T1, T2 = 3 * PER, 7 * PER
WIN = np.where((T >= T1) & (T <= T2), 1.0, 0.0)


def make_plot(signals_dict, title="PQ Disturbance Gallery"):
    n = len(signals_dict)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    palette = plt.cm.tab10.colors
    for ax, (lbl, sig), col in zip(axes, signals_dict.items(), palette):
        ax.plot(T * 1000, sig, color=col, linewidth=1.1)
        ax.set_title(lbl, fontsize=9, fontweight='bold')
        ax.set_ylabel("Amp (pu)", fontsize=8)
        ax.axvline(T1 * 1000, color='gray', lw=0.8, linestyle='--')
        ax.axvline(T2 * 1000, color='gray', lw=0.8, linestyle='--')
        ax.grid(True, linestyle=':', alpha=0.5)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


# ── 1. Pure Sinusoid ──────────────────────────────────────────────────────────
pure = np.sin(OMEGA * T)

# ── 2. Voltage Sag ───────────────────────────────────────────────────────────
alpha_sag = 0.5
sag = (1.0 - alpha_sag * WIN) * np.sin(OMEGA * T)

# ── 3. Voltage Swell ──────────────────────────────────────────────────────────
alpha_swell = 0.4
swell = (1.0 + alpha_swell * WIN) * np.sin(OMEGA * T)

# ── 4. Interruption ───────────────────────────────────────────────────────────
alpha_int = 0.95
interruption = (1.0 - alpha_int * WIN) * np.sin(OMEGA * T)

# ── 5. Harmonics (3rd, 5th, 7th) ─────────────────────────────────────────────
harmonics = (np.sin(OMEGA * T)
             + 0.15 * np.sin(3 * OMEGA * T)
             + 0.10 * np.sin(5 * OMEGA * T)
             + 0.05 * np.sin(7 * OMEGA * T))

# ── 6. Flicker ────────────────────────────────────────────────────────────────
lambda_f  = 0.15
f_flicker = 8.0
flicker = (1.0 + lambda_f * np.sin(2 * np.pi * f_flicker * T)) * np.sin(OMEGA * T)

# ── 7. Oscillatory Transient (damped sinusoid injection) ─────────────────────
t_trans    = T1
tau_t      = 0.015
f_trans    = 550.0
trans_mask = (T >= t_trans)
transient  = np.sin(OMEGA * T)
transient[trans_mask] += (0.45
                          * np.exp(-(T[trans_mask] - t_trans) / tau_t)
                          * np.sin(2 * np.pi * f_trans * (T[trans_mask] - t_trans)))

# ── 8. Voltage Notch (sub-cycle, 10-cycle window) ────────────────────────────
T_notch = np.linspace(0, 10 * PER, int(FS * 10 * PER), endpoint=False)
v_notch = np.sin(OMEGA * T_notch)
notch_d = 0.4
notch_w = 0.05    # radians

for c in range(10):
    for phase_deg in [30, 210]:
        t_start = c * PER + np.radians(phase_deg) / OMEGA
        t_end   = t_start + notch_w / OMEGA
        mask    = (T_notch >= t_start) & (T_notch <= t_end)
        v_notch = np.where(mask, v_notch - notch_d * np.sign(v_notch), v_notch)

# ── 9. Hybrid: Sag + Harmonics ───────────────────────────────────────────────
sag_harm = ((1.0 - 0.4 * WIN) * np.sin(OMEGA * T)
            + 0.15 * np.sin(3 * OMEGA * T) * WIN
            + 0.10 * np.sin(5 * OMEGA * T) * WIN
            + 0.05 * np.sin(7 * OMEGA * T) * WIN)

# ── 10. Hybrid: Swell + Harmonics ────────────────────────────────────────────
swell_harm = ((1.0 + 0.4 * WIN) * np.sin(OMEGA * T)
              + 0.12 * np.sin(3 * OMEGA * T) * WIN
              + 0.06 * np.sin(5 * OMEGA * T) * WIN)

# ── 11. Hybrid: Flicker + Sag ────────────────────────────────────────────────
flicker_sag = ((1.0 - 0.4 * WIN)
               * (1.0 + 0.15 * np.sin(2 * np.pi * 8 * T))
               * np.sin(OMEGA * T))

# ── 12. Hybrid: Swell + Oscillatory Transient ────────────────────────────────
t3, t4   = T1 + 2 * PER, T1 + 4 * PER
tmask2   = np.where((T >= t3) & (T <= t4), 1.0, 0.0)
swell_trans = ((1.0 + 0.3 * WIN) * np.sin(OMEGA * T)
               + (0.4 * np.exp(-(T - t3) / 0.01)
                  * np.sin(2 * np.pi * 600 * (T - t3))) * tmask2)

# ── Plot all single disturbances ──────────────────────────────────────────────
make_plot({
    "1. Pure Sinusoid (Normal)":          pure,
    "2. Voltage Sag (alpha=0.5)":         sag,
    "3. Voltage Swell (alpha=0.4)":       swell,
    "4. Voltage Interruption (alpha=0.95)": interruption,
    "5. Harmonics (3rd/5th/7th)":         harmonics,
    "6. Voltage Flicker (ff=8 Hz)":       flicker,
    "7. Oscillatory Transient (550 Hz)":  transient,
}, title="Single Disturbance Classes")

# ── Plot hybrid disturbances ──────────────────────────────────────────────────
make_plot({
    "8.  Sag + Harmonics":                sag_harm,
    "9.  Swell + Harmonics":              swell_harm,
    "10. Flicker + Sag":                  flicker_sag,
    "11. Swell + Oscillatory Transient":  swell_trans,
}, title="Hybrid / Compound Disturbance Classes")

# ── Polar Trajectory Circles (Paper 5 approach) ───────────────────────────────
def polar_plot(sig, label, color, ax):
    analytic = hilbert(sig)
    R     = np.abs(analytic)
    Theta = np.angle(analytic)
    ax.plot(Theta, R, color=color, lw=0.8, alpha=0.85)
    ax.set_title(label, fontsize=8, pad=10)
    ax.set_rmax(2.0)
    # Fix: place radial tick labels at 67.5 degrees to avoid overlap
    ax.set_rlabel_position(67.5)
    # Fix: reduce number of radial ticks shown
    ax.set_rticks([0.5, 1.0, 1.5, 2.0])
    # Fix: smaller font for radial labels
    ax.tick_params(axis='y', labelsize=6)
    # Fix: lighter grid so labels are readable
    ax.grid(True, linestyle=':', alpha=0.4)


fig2, traj_axes = plt.subplots(2, 4, figsize=(16, 8),
                               subplot_kw={'projection': 'polar'})

traj_signals = [
    (pure,        "Normal (perfect circle)",  "teal"),
    (sag,         "Sag (inner ring)",          "blue"),
    (swell,       "Swell (outer ring)",        "green"),
    (interruption,"Interruption (collapsed)",  "gray"),
    (harmonics,   "Harmonics (rippled)",       "purple"),
    (flicker,     "Flicker (breathing)",       "orange"),
    (sag_harm,    "Sag + Harmonics",           "crimson"),
    (swell_harm,  "Swell + Harmonics",         "darkred"),
]

for ax, (sig, lbl, col) in zip(traj_axes.flat, traj_signals):
    polar_plot(sig, lbl, col, ax)

fig2.suptitle("Polar Trajectory Circles - Visual Signature per Disturbance Class",
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
 

print("\nGeneration complete. All unique disturbance classes simulated.")
print(f"Sampling rate: {FS} Hz | Duration: {DURATION*1000:.0f} ms | Samples: {len(T)}")
