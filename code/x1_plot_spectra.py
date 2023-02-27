import os
import argparse

import pandas as pd
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

csv = pd.read_csv(args.csv, skiprows=2, delimiter=",")
if csv.shape[1] > 4:
    csv.columns = ["wl1", "ab1","wl2", "ab2","wl3", "ab3"]
else:
    csv.columns = ["wl1", "ab1","wl3", "ab3"]

fig, ax = plt.subplots(figsize=(3.25, 2.2))

df = csv[["wl1", "ab1"]].sort_values("wl1")
ax.plot(df.wl1, df.ab1, 'k-', label = "GNR in H2O")

df = csv[["wl3", "ab3"]].sort_values("wl3")
ax.plot(df.wl3, df.ab3, 'g-', label = "After purification")

if "wl2" in csv:
    df = csv[["wl2", "ab2"]].sort_values("wl2")
    ax.plot(df.wl2, df.ab2, 'r-', label = "After overcoating")

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
