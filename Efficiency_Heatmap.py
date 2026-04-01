import numpy as np
import matplotlib.pyplot as plt
from Voltage_simulation import current_from_bandgaps, voltage, calculate_ff

# Finding the current for a range of bandgaps to find optimal, maximum current when top current = bottom current

def efficieny_from_bandgap(top_bandgap, bottom_bandgap):

    current = current_from_bandgaps(top_bandgap, bottom_bandgap)
    current_matched = np.minimum(current[0], current[1])
    voltages = voltage(top_bandgap, bottom_bandgap)

    # Inside your loop:
    ff_top = calculate_ff(voltages[0])
    ff_bot = calculate_ff(voltages[1])

    # The "Working" voltage
    v_working = (voltages[0] * ff_top) + (voltages[1] * ff_bot)

    can_split_water = (v_working >= 1.8)

    eff = ((1.23 * current_matched) / 100) * can_split_water  # only 1.23 V efficiently used to split water

    return eff


top_gaps = np.arange(0.7, 3, 0.02)    # Testing every 0.05 eV
bottom_gaps = np.arange(0.7, 3, 0.02)

results_grid = np.zeros((len(top_gaps), len(bottom_gaps)))

for i, top in enumerate(top_gaps):
    for j, bottom in enumerate(bottom_gaps):
        results_grid[i,j] = efficieny_from_bandgap(top, bottom)

# Finding the ridge line (best top bandgap for each bottom bandgap)
ridge_indices = np.argmax(results_grid, axis = 0)
ridge_top = top_gaps[ridge_indices]
ridge_bottom = bottom_gaps

mask = np.max(results_grid, axis=0) > 0
ridge_top = ridge_top[mask]
ridge_bottom = ridge_bottom[mask]

# Creating the heatmap plot
plt.figure(figsize=(8, 6))
img = plt.imshow(results_grid,
           extent=[bottom_gaps.min(), bottom_gaps.max(), top_gaps.min(), top_gaps.max()],
           origin='lower',
           aspect='auto',  # This makes the square fill the plot area
           cmap='magma')  # 'magma' or 'viridis' both look great for heatmaps


real_cells = [
    [1.12, 1.68, 'Perovskite/Si (Record)'],
    [1.12, 1.72, 'III-V/Si'],
    [1.22, 1.75, 'All-Perovskite'],
    [1.04, 1.63, 'CIGS Tandem']
]

# Plot each one
for bot, top, label in real_cells:
    plt.scatter(bot, top, edgecolor='white', s=80, label=label)

plt.plot(ridge_bottom, ridge_top, color='lavender', linestyle='-', linewidth=2, label='Optimal Ridge')

plt.colorbar(img, label='STH Efficiency %')
plt.xlabel('Bottom Bandgap (eV)')
plt.ylabel('Top Bandgap (eV)')
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.title('Tandem Solar-to-Hydrogen Efficiency Map')
plt.show()

#Find max efficiency and bandgap combination

print(f"The maximum efficiency is {round(np.max(results_grid) *100, 2) }%")
best_top_bandgap,best_bottom_bandgap = np.unravel_index(np.argmax(results_grid), results_grid.shape)
print(f"The best bottom bandgap is {round(bottom_gaps[best_bottom_bandgap], 2)} eV")
print(f"The best top bandgap is {round(top_gaps[best_top_bandgap], 2)} eV")
