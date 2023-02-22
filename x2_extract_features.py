import os
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
try:
    plt.style.use("matplotlib.mplstyle")
except:
    pass

parser = argparse.ArgumentParser(
    description = 'Extract spectral features into a CSV file',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('csv', help="CSV file containing spectra csv file names")

def extract_peak_fwhm(x, y, plot=False):
    # BSpline Fit
    bspl = make_interp_spline(x, y, k=3)
    xx, dx = np.linspace(400, 950, 51, retstep = True)

    if plot:
        # Plot the spline fit to make sure
        plt.plot(xx, bspl(xx), 'r-', label="BSpline")
        plt.plot(x, y, 'k-', label="Original")
        plt.legend()
        plt.xticks(range(400, 950, 100))
        plt.grid()
        plt.show()

    pks = find_peaks(bspl(xx))[0]
    widths = peak_widths(bspl(xx), pks, rel_height = 0.5)[0]
    fwhm = widths * dx

    return xx[pks], fwhm


# Parse Arguments
args = parser.parse_args()

# Create output directory and set output file
os.makedirs("Data", exist_ok=True)
out = "Data/%s" %(os.path.basename(args.csv))

# Read the index CSV file
print("Reading", args.csv, "...")
csv = pd.read_csv(args.csv)

# To dynamically generate a pandas dataframe
df = {
    'name': [],
    'tspk1': [],
    'tsfw1': [],
    'lspk1': [],
    'lsfw1': [],
    'tspk2': [],
    'tsfw2': [],
    'lspk2': [],
    'lsfw2': [],
    'tspk3': [],
    'tsfw3': [],
    'lspk3': [],
    'lsfw3': [],
}

# Loop over the spectra files
for fname in csv.name:
    df['name'].append(fname)

    # Read the spectra
    fpath = os.path.join(os.path.dirname(args.csv), fname+".csv")
    print("Read", fpath, end = "... ")
    spec = np.loadtxt(fpath, skiprows=3, delimiter=",")

    # In water
    pk, fw = extract_peak_fwhm(spec[:, 0], spec[:, 1])
    df['tspk1'].append(pk[0])
    df['lspk1'].append(pk[1])
    df['tsfw1'].append(fw[0])
    df['lsfw1'].append(fw[1])

    # After purification
    pk, fw = extract_peak_fwhm(spec[:, -2], spec[:, -1])
    df['tspk3'].append(pk[0])
    df['lspk3'].append(pk[1])
    df['tsfw3'].append(fw[0])
    df['lsfw3'].append(fw[1])

    # After overcoating
    if spec.shape[1] > 4:
        pk, fw = extract_peak_fwhm(spec[:, 2], spec[:, 3])
        df['tspk2'].append(pk[0])
        df['lspk2'].append(pk[1])
        df['tsfw2'].append(fw[0])
        df['lsfw2'].append(fw[1])
    else:
        df['tspk2'].append("")
        df['lspk2'].append("")
        df['tsfw2'].append("")
        df['lsfw2'].append("")

    print("OK")

pd.DataFrame(df).to_csv(out, index=False)
print("Save OK: %s" %out)
