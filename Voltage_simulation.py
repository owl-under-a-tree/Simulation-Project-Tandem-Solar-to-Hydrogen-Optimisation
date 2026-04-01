from scipy.integrate import quad
import numpy as np
from sun_data import data, current_from_data

h = 6.63e-34
c = 2.998e8
k = 1.38e-23
q = 1.6e-19

top_bandgap = 2.3
bottom_bandgap = 1.3

def j0_integrand(E, T):
    numerator = E ** 2
    denominator = (c ** 2 * h ** 3) * (np.exp(E / (k * T)) - 1)
    return numerator / denominator

def J0(bandgap, T=300):
    energy = bandgap * q
    upper_limit = energy + 50 * k * T  # Integrand negligible beyond ~50 kT
    photon_flux, _ = quad(j0_integrand, energy, upper_limit, args=(T,))

    j0_amp_m2 = 2 * np.pi * q * photon_flux

    # Divide by 10 to get mA/cm^2
    return j0_amp_m2 / 10

def current_from_bandgaps(top_bandgap, bottom_bandgap):
    bottom_cell_data = data.loc[
        ((data["Energy/J"] < (top_bandgap * 1.6e-19)) & (data["Energy/J"] > (bottom_bandgap * 1.6e-19))), [
            "Wavelength", "Photon Flux / m^-2 nm^-1", "Energy/J"]]
    top_cell_data = data.loc[
        data["Energy/J"] >= (top_bandgap * 1.6e-19), ["Wavelength", "Photon Flux / m^-2 nm^-1", "Energy/J"]]
    top_current = current_from_data(top_cell_data)
    bottom_current = current_from_data(bottom_cell_data)
    current_data = [top_current, bottom_current]
    return current_data

def voltage(top_bandgap, bottom_bandgap):
    top_J0 = J0(top_bandgap, T = 300)
    bottom_J0 = J0(bottom_bandgap, T = 300)

    j_sc = current_from_bandgaps(top_bandgap, bottom_bandgap)
    thermal_voltage = (k*300)/q

    top_voltage = thermal_voltage*(np.log((j_sc[0] / top_J0) + 1))
    bottom_voltage = thermal_voltage*(np.log((j_sc[1]/bottom_J0) + 1))
    voltage = [top_voltage, bottom_voltage]

    return voltage

def calculate_ff(voc, T=300):
    # Thermal voltage (approx 0.02585 V at 300K)
    vt = (k * T) / q

        # Normalized voltage (dimensionless)
    v_norm = voc / vt

        # Green's Approximation
    ff = (v_norm - np.log(v_norm + 0.72)) / (v_norm + 1)
    return ff
