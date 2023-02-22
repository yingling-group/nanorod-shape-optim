import os
import argparse

import numpy as np 
import matplotlib.pyplot as plt 

try:
    plt.style.use("matplotlib.mplstyle")
except:
    pass

parser = argparse.ArgumentParser(
    description = 'Save spectra data from CSV file to a plot',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('csv')
args = parser.parse_args()

os.makedirs("Plots", exist_ok=True)
out = "Plots/%s.png" %(os.path.basename(args.csv))

print("Plotting", args.csv, "...")

csv = np.loadtxt(args.csv, skiprows=3, delimiter=",")

fig, ax = plt.subplots(figsize=(3.25, 2.2))
ax.plot(csv[:,0], csv[:,1], 'k-', label = "GNR in H2O")
ax.plot(csv[:,2], csv[:,3], 'r-', label = "After overcoating")
ax.plot(csv[:,4], csv[:,5], 'g-', label = "After purification")

plt.xlim(400, 1000)
plt.ylim(0, 2)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (a.u.)")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(out, dpi=600)
plt.show()

print("Save OK: %s" %out)
