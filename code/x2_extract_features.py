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

# Read Melanie and Rowe CSV files one by one from 'Data/Spectra/'
# Manually combine them to 'Data/all_spectra.csv'

parser = argparse.ArgumentParser(
    description = 'Extract spectral features into a CSV file',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('csv', help="CSV file containing spectra csv file names")
parser.add_argument('--plot', help="Save the spectra as a plot", action="store_true")
parser.add_argument('--show', help="Display spline fit and found peak positions", action="store_true")

def plot_spectra(csvfile):
    os.makedirs("Plots", exist_ok=True)
    out = "Plots/%s.png" %(os.path.basename(csvfile))

    print("Plotting", csvfile, "...")

    csv = pd.read_csv(csvfile, skiprows=2, delimiter=",")
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
    plt.close()

    print("Save OK: %s" %out)


def extract_peak_fwhm(df):
    df.columns = ["wl", "ab"]
    df = df.sort_values("wl").dropna()
    x = df["wl"]
    y = df["ab"]

    # BSpline Fit
    bspl = make_interp_spline(x, y, k=3)
    xx, dx = np.linspace(400, 900, 501, retstep = True)

    pks = find_peaks(bspl(xx), distance=100, prominence=0.02)[0]
    widths = peak_widths(bspl(xx), pks, rel_height = 0.5)[0]
    fwhm = widths * dx

    try:
        tspk = xx[pks[-2]]
        lspk = xx[pks[-1]]    
        tsfw = fwhm[-2]
        lsfw = fwhm[-1]

        if args.show:
            # Plot the spline fit to make sure
            plt.figure(dpi=300, figsize=(3.25, 2.2))
            plt.plot(xx, bspl(xx), 'r-', label="BSpline")
            plt.plot(x, y, 'k-', label="Original")
            plt.plot(tspk, bspl(tspk), 'g*', label="TSPK")
            plt.plot(lspk, bspl(lspk), 'b*', label="LSPK")

            plt.legend()
            plt.xticks(range(400, 900, 100))
            plt.xlim(400, 900)
            plt.grid()
            plt.show()
    except IndexError:
        print("failed to find the peaks", end=" ... ")
        if args.show:
            # Plot the spline fit to make sure
            plt.figure(dpi=300, figsize=(3.25, 2.2))
            plt.plot(xx, bspl(xx), 'r-', label="BSpline")
            plt.plot(x, y, 'k-', label="Original")
            plt.title("Failed to find peaks")
            plt.legend()
            plt.xticks(range(400, 900, 100))
            plt.xlim(400, 900)
            plt.grid()
            plt.show()
        return "", "", "", ""

    return tspk, lspk, tsfw, lsfw


def calc_lobe_fraction(df):
    # teosVolume,teosVolPct,thickness,quality,rodssquare,rodsfull,rodslobe,rodshalf,rodsagg,rodsline
    tot = df.rodssquare + df.rodsfull + df.rodslobe + df.rodshalf + df.rodsagg + df.rodsline
    return df.rodslobe / tot

def calc_full_fraction(df):
    # teosVolume,teosVolPct,thickness,quality,rodssquare,rodsfull,rodslobe,rodshalf,rodsagg,rodsline
    tot = df.rodssquare + df.rodsfull + df.rodslobe + df.rodshalf + df.rodsagg + df.rodsline
    return df.rodsfull / tot

def calc_other_fraction(df):
    # teosVolume,teosVolPct,thickness,quality,rodssquare,rodsfull,rodslobe,rodshalf,rodsagg,rodsline
    tot = df.rodssquare + df.rodsfull + df.rodslobe + df.rodshalf + df.rodsagg + df.rodsline
    other = df.rodssquare + df.rodshalf + df.rodsagg + df.rodsline
    return other / tot

# def calc_quality(df):
#     return df[["rodssquare","rodsfull","rodslobe","rodshalf","rodsagg","rodsline"]].std(axis=1)

def add_column(dfdict, csv, col):
    if col in csv:
        dfdict[col] = csv[col]
    return dfdict

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

    if args.plot:
        plot_spectra(fpath)

    print("Read", fpath, end = " ... ")
    spec = pd.read_csv(fpath, skiprows=2, delimiter=",")

    # set column names
    if spec.shape[1] > 4:
        spec.columns = ["wl1", "ab1","wl2", "ab2","wl3", "ab3"]
    else:
        spec.columns = ["wl1", "ab1","wl3", "ab3"]

    # In water
    tspk, lspk, tsfw, lsfw = extract_peak_fwhm(spec[["wl1", "ab1"]])
    df['tspk1'].append(tspk)
    df['lspk1'].append(lspk)
    df['tsfw1'].append(tsfw)
    df['lsfw1'].append(lsfw)

    # After purification
    tspk, lspk, tsfw, lsfw = extract_peak_fwhm(spec[["wl3", "ab3"]])
    df['tspk3'].append(tspk)
    df['lspk3'].append(lspk)
    df['tsfw3'].append(tsfw)
    df['lsfw3'].append(lsfw)

    # After overcoating
    if "wl2" in spec:
        tspk, lspk, tsfw, lsfw = extract_peak_fwhm(spec[["wl2", "ab2"]])
        df['tspk2'].append(tspk)
        df['lspk2'].append(lspk)
        df['tsfw2'].append(tsfw)
        df['lsfw2'].append(lsfw)
    else:
        df['tspk2'].append("")
        df['lspk2'].append("")
        df['tsfw2'].append("")
        df['lsfw2'].append("")

    print("OK")

# Add additional variables if exist
df = add_column(df, csv, 'teosVolume')
df = add_column(df, csv, 'teosVolPct')

try:
    # Training dataset
    df['quality']    = csv.quality
    df['lobe']    = calc_lobe_fraction(csv)
    df['full']    = calc_full_fraction(csv)
    df['other']   = calc_other_fraction(csv)
except:
    # Testing dataset
    df = add_column(df, csv, 'lobe')
    df = add_column(df, csv, 'full')
    df = add_column(df, csv, 'other')

# Save
pd.DataFrame(df).to_csv(out, index=False)
print("Save OK: %s" %out)
