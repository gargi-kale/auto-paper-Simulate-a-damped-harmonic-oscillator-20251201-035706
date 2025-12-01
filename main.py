# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0  # mass (kg)
k = 1.0  # spring constant (N/m)
c_crit = 2.0 * np.sqrt(k * m)  # critical damping coefficient

# Time settings
t_start = 0.0
t_end = 20.0
dt = 0.01
t = np.arange(t_start, t_end + dt, dt)

# Initial conditions
x0 = 1.0
v0 = 0.0

def rhs(state, c):
    """Return derivatives [dx/dt, dv/dt] for given state and damping c."""
    x, v = state
    dxdt = v
    dvdt = -(c / m) * v - (k / m) * x
    return np.array([dxdt, dvdt])

def rk4_step(state, c, h):
    k1 = rhs(state, c)
    k2 = rhs(state + 0.5 * h * k1, c)
    k3 = rhs(state + 0.5 * h * k2, c)
    k4 = rhs(state + h * k3, c)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(c):
    """Simulate the damped oscillator for a given damping coefficient c.
    Returns displacement array x(t)."""
    state = np.array([x0, v0])
    xs = np.empty_like(t)
    for i, _ in enumerate(t):
        xs[i] = state[0]
        state = rk4_step(state, c, dt)
    return xs

def extract_decay_constant(c, x):
    """Estimate decay constant from peak amplitudes of displacement."""
    # Find local maxima (simple discrete check)
    peaks_idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    if len(peaks_idx) < 2:
        # Not enough peaks (over‑damped or critically damped); fall back to analytical value
        return c / (2.0 * m)
    t_peaks = t[peaks_idx]
    amp_peaks = np.abs(x[peaks_idx])
    # Linear fit to log(amplitude) = -lambda * t + const
    coeffs = np.polyfit(t_peaks, np.log(amp_peaks), 1)
    decay_const = -coeffs[0]
    return decay_const

# ------------------------------------------------------------
# Experiment 1: Damping regimes time response
# ------------------------------------------------------------
c_under = 0.5 * c_crit   # under‑damped
c_critical = c_crit      # critically damped
c_over = 2.0 * c_crit    # over‑damped

x_under = simulate(c_under)
x_critical = simulate(c_critical)
x_over = simulate(c_over)

plt.figure(figsize=(8, 5))
plt.plot(t, x_under, label=f'Underdamped (c={c_under:.2f})')
plt.plot(t, x_critical, label=f'Critical (c={c_critical:.2f})')
plt.plot(t, x_over, label=f'Overdamped (c={c_over:.2f})')
plt.title('Displacement vs Time for Different Damping Regimes')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('displacement_vs_time_regimes.png')
plt.close()

# ------------------------------------------------------------
# Experiment 2: Decay rate vs damping coefficient (extracted from simulation)
# ------------------------------------------------------------
c_vals = np.linspace(0.1, 5.0, 200)
decay_constants = np.empty_like(c_vals)

for i, c_val in enumerate(c_vals):
    x_sim = simulate(c_val)
    decay_constants[i] = extract_decay_constant(c_val, x_sim)

plt.figure(figsize=(8, 5))
plt.plot(c_vals, decay_constants, color='tab:blue')
plt.title('Decay Constant vs Damping Coefficient (extracted from peaks)')
plt.xlabel('Damping Coefficient c (N·s/m)')
plt.ylabel('Decay Constant (1/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig('decay_constant_vs_damping.png')
plt.close()

# ------------------------------------------------------------
# Final answer: critical damping coefficient value
# ------------------------------------------------------------
print('Answer:', c_crit)
