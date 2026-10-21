"""
Pluto Barycentric Radial Velocity Simulation and Signal Analysis.

This module models the gravitational wobble of Pluto induced by its moons
and validates the extracted orbital frequencies using Fourier analysis.
"""


import numpy as np
import scipy.fft as fft  # type: ignore
import scipy.signal.windows as windows  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import astropy.units as u  # type: ignore
import astropy.constants as const  # type: ignore
from typing import Any, Tuple, List


def simulate_pluto_wobble(t: Any, noise_amplitude: Any = 1.5 * u.m / u.s) -> Any:
    """
    Simulates Pluto's radial velocity line-of-sight wobble over time.

    Parameters
    ----------
    t : astropy.units.Quantity
        Time array array for the orbital simulation.
    noise_amplitude : astropy.units.Quantity, optional
        Standard deviation of the Gaussian noise added to the signal.

    Returns
    -------
    astropy.units.Quantity
        Combined radial velocity signal containing moons and random noise.
    """
    period_charon = 6.387 * u.day
    period_hydra = 38.20 * u.day

    v_amp_charon = 24.2 * u.m / u.s
    v_amp_hydra = 0.8 * u.m / u.s

    signal_charon = v_amp_charon * np.sin(2 * np.pi * (t / period_charon).to(u.dimensionless_unscaled).value)
    signal_hydra = v_amp_hydra * np.sin(2 * np.pi * (t / period_hydra).to(u.dimensionless_unscaled).value)

    np.random.seed(42) 
    noise = np.random.normal(0, noise_amplitude.value, len(t)) * noise_amplitude.unit

    return signal_charon + signal_hydra + noise


def analyze_frequencies(t: Any, velocity: Any, dt_scalar: Any = 0.1 * u.day) -> Tuple[Any, Any, Any]:
    """
    Applies a windowing function and processes the 1D forward Fast Fourier Transform.

    Parameters
    ----------
    t : astropy.units.Quantity
        Time step vector arrays tracking the simulation window.
    velocity : astropy.units.Quantity
        Simulated multi-body velocity signal data.
    dt_scalar : astropy.units.Quantity, optional
        Discrete time intervals separating measurement observations.

    Returns
    -------
    freqs : astropy.units.Quantity
        Array of parsed spatial frequency components.
    power_spectral_density : numpy.ndarray
        Calculated spectral intensity values mapping dominant frequencies.
    raw_fft : numpy.ndarray
        Unscaled complex array values tracking raw forward matrix steps.
    """

    n = len(t)

    window_profile = windows.hann(n)
    windowed_velocity = velocity.value * window_profile

    raw_fft = fft.fft(windowed_velocity)

    raw_freqs = fft.fftfreq(n, d=dt_scalar.to(u.day).value)
    freqs = raw_freqs * (1 / u.day)
    power_spectral_density = np.abs(raw_fft) ** 2 / n

    return freqs, power_spectral_density, raw_fft


def reconstruct_time_signal(raw_fft: Any) -> Any:
    """
    Performs an Inverse Fast Fourier Transform (IFFT).

    Parameters
    ----------
    raw_fft : numpy.ndarray
        Complex multidimensional array tracking forward spectral operations.

    Returns
    -------
    numpy.ndarray
        Reconstructed real-valued velocity curves converted to time-space.
    """

    reconstructed_complex = fft.ifft(raw_fft)
    reconstructed_velocity = np.real(reconstructed_complex)

    return reconstructed_velocity


def verify_with_keplers_third_law(detected_frequency: Any, semi_major_axis: Any) -> bool:
    """
    Compares an FFT frequency peak with Kepler's Third Law.

    Parameters
    ----------
    detected_frequency : astropy.units.Quantity
        Numerical target frequency spike mapped by the forward transform.
    semi_major_axis : astropy.units.Quantity
        Physical orbital distance tracking system moons.

    Returns
    -------
    bool
        True if the findings match physical constants within limits, else False.
    """

    m_total = 1.46e22 * u.kg
    analytical_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (const.G * m_total))
    analytical_freq = (1.0 / analytical_period).to(1 / u.day)

    print("\nVerification Check: ")
    print(f"Kepler's Analytical Frequency: {analytical_freq:.4f}")
    print(f"FFT Numerically Detected Peak: {detected_frequency:.4f}")

    is_correct = bool(np.isclose(detected_frequency.value, analytical_freq.value, rtol=1.5e-2))
    print(f"Verification Check Status:      {'Passed' if is_correct else 'Failure'}")

    return is_correct


def animate_pluto_wobble() -> None:
    """
    Generates a 2D orbital animation of the Pluto-Charon binary system.
    """

    period = 6.387
    omega = 2 * np.pi / period
    r_pluto = 0.2  
    r_charon = 1.0

    fig, (ax_orbit, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))

    ax_orbit.set_xlim(-1.5, 1.5)
    ax_orbit.set_ylim(-1.5, 1.5)
    ax_orbit.set_aspect('equal')
    ax_orbit.grid(True, linestyle=':')
    ax_orbit.set_title("Orbital View (Barycenter Origin)")
    ax_orbit.set_xlabel("x Position")
    ax_orbit.set_ylabel("y Position")

    ax_orbit.plot(0, 0, 'kx', markersize=10, label='Barycenter')
    pluto_dot, = ax_orbit.plot([], [], 'o', color='darkblue', markersize=12, label='Pluto')
    charon_dot, = ax_orbit.plot([], [], 'o', color='gray', markersize=6, label='Charon')
    ax_orbit.legend(loc='upper right')

    time_history: List[float] = []
    velocity_history: List[float] = []
    rv_line, = ax_plot.plot([], [], color='crimson', linewidth=2)
    ax_plot.set_xlim(0, 10)
    ax_plot.set_ylim(-30, 30)
    ax_plot.set_title("Observed Radial Velocity (Wobble)")
    ax_plot.set_xlabel("Time (Days)")
    ax_plot.set_ylabel("Velocity (m/s)")
    ax_plot.grid(True, linestyle=':')

    def update(frame: int) -> Tuple[Any, Any, Any]:
        t = frame * 0.1

        if frame == 0:
            time_history.clear()
            velocity_history.clear()
            ax_plot.set_xlim(0, 10)

        x_pluto = r_pluto * np.cos(omega * t)
        y_pluto = r_pluto * np.sin(omega * t)
        x_charon = -r_charon * np.cos(omega * t)
        y_charon = -r_charon * np.sin(omega * t)

        pluto_dot.set_data([x_pluto], [y_pluto])
        charon_dot.set_data([x_charon], [y_charon])

        v_radial = 24.2 * np.cos(omega * t)
        time_history.append(t)
        velocity_history.append(v_radial)

        if t > ax_plot.get_xlim()[1]:
            ax_plot.set_xlim(t - 10, t)
 
        rv_line.set_data(time_history, velocity_history)
        return pluto_dot, charon_dot, rv_line

    global animate
    animate = FuncAnimation(fig, update, frames=np.arange(0, 200), interval=50, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Executes parameters, coordinates transforms, and launches visualizations sequentially.
    """

    total_duration = 120 * u.day
    sampling_interval = 0.1 * u.day
    t = np.arange(0, total_duration.value, sampling_interval.value) * u.day

    velocity_data = simulate_pluto_wobble(t)
    freqs, psd, raw_fft = analyze_frequencies(t, velocity_data, dt_scalar=sampling_interval)

    positive_mask = freqs.value > 0
    pos_freqs = freqs[positive_mask]
    pos_psd = psd[positive_mask]

    peak_idx = np.argmax(pos_psd)
    detected_charon_freq = pos_freqs[peak_idx]

    charon_semi_major_axis = 19596 * u.km
    verify_with_keplers_third_law(detected_charon_freq, charon_semi_major_axis)

    reconstructed_velocity = reconstruct_time_signal(raw_fft)

    print("\nInverse Fourier Transform: ")
    print(f"{'Time (days)':<15} | {'Reconstructed Velocity (m/s)':<30}")
    print("-" * 50)
    for time_val, vel_val in zip(t.value[:10], reconstructed_velocity[:10]):
        print(f"{time_val:<15.4f} | {vel_val:<30.4f}")
    print("-" * 50)
    print(f"Showing the first 10 of {len(t)} total discrete data points.\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(t.value, velocity_data.value, color='darkblue', alpha=0.7, label='Observed (With Noise)')
    ax1.set_xlabel(f"Time ({t.unit})")
    ax1.set_ylabel(f"Radial Velocity ({velocity_data.unit})")
    ax1.set_title("Pluto Barycentric Radial Velocity")
    ax1.grid(True, linestyle='--')
    ax1.legend()
    ax2.plot(pos_freqs.value, pos_psd, color='crimson', linewidth=2, label='Power Spectrum')
    ax2.axvline(detected_charon_freq.value, color='black', linestyle=':', label=f'Charon Peak ({detected_charon_freq:.3f})')
    ax2.set_xlim(0, 0.3)
    ax2.set_xlabel("Frequency (1/day)")
    ax2.set_ylabel("Spectral Power")
    ax2.set_title("Frequency Spectrum (PSD)")
    ax2.grid(True, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("\nLaunching orbital simulation animation tracker...")
    animate_pluto_wobble()


if __name__ == '__main__':
    main()
