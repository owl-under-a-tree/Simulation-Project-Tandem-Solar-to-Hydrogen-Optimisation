import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sunlight.csv', skiprows = 1) # Import the spectral data
data = df.loc[: , ["Wvlgth nm", "Global tilt  W*m-2*nm-1"]].copy()

h = 6.63e-34
c = 2.998e8


data.columns = ["Wavelength", "Irradiance"]
data = data[data["Wavelength"] <= 1200]
data['Energy/J'] = (h*c)/(data.iloc[:, 0] * 1e-9)
data["Photon Flux / m^-2 nm^-1"] = data["Irradiance"]/data["Energy/J"] # Photon flux is number of photons per second per unit area per nm

def current_from_data(data):
    total_photons_per_second = np.trapezoid(data["Photon Flux / m^-2 nm^-1"], x = data["Wavelength"])
    total_current = total_photons_per_second*1.60218e-19
    current_mAcm2 = total_current/10
    return current_mAcm2

# Plotting the sun data code

if __name__ == "__main__":

    material_bandgap = 1.6 * 1.60218e-19 # Change the material bandgap in eV to whatever is appropriate
    bottom_cell_data = data.loc[data["Energy/J"] < material_bandgap, ["Wavelength", "Photon Flux / m^-2 nm^-1","Energy/J"]]
    top_cell_data = data.loc[data["Energy/J"] >= material_bandgap, ["Wavelength", "Photon Flux / m^-2 nm^-1","Energy/J"]]

    # Calculating the maximum current of the bottom cell - system current must be the same and therefore system is limited by the current achievable from the cells
    print(f"Total possible solar current = {current_from_data(data)} ")
    print(f"Total possible current from top cell = {current_from_data(top_cell_data)}")
    print(f"Total possible current from bottom cell = {current_from_data(bottom_cell_data)}")


    plt.figure(figsize=(10, 6))

    # Plotting the Photon Flux vs Wavelength
    plt.plot(data["Wavelength"], data["Photon Flux / m^-2 nm^-1"], color='black', label='Photon Flux')

    plt.fill_between(data["Wavelength"], data["Photon Flux / m^-2 nm^-1"],
                     where=(data["Wavelength"] >= 400) & (data["Wavelength"] <= 700),
                     color='green', alpha=0.3, label='Visible Range')

    # The bottom tandem cell only sees the photons not harvested by the top cell (to a good approximation)

    plt.fill_between(data["Wavelength"], data["Photon Flux / m^-2 nm^-1"], where = (data["Energy/J"] >= material_bandgap),
                     color = "cyan", alpha = 0.3, label = "Top cell harvesting region")

    #
    plt.fill_between(data["Wavelength"], data["Photon Flux / m^-2 nm^-1"], where = (data["Energy/J"] < material_bandgap),
                     color = "red", alpha = 0.1, label = "Possible bottom cell harvesting region")

    material_bandgap_wavelength = (h*c)/material_bandgap * 1e9

    plt.axvline(x = material_bandgap_wavelength, ymin = 0, ymax = 5, color = 'blue',
                label = "Cuttoff wavelength")

    # Formatting the chart
    plt.title("Solar Photon Flux Spectrum", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Flux (photons s$^{-1}$ m$^{-2}$ nm$^{-1}$)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
