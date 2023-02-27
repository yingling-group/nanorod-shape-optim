# %%
import argparse
import pandas as pd 
from model import plotlib, features

import matplotlib.pyplot as plt 
try:
    plt.style.use("matplotlib.mplstyle")
except:
    pass

parser = argparse.ArgumentParser(
    description = 'Plot and save correlation and distribution of features',
    epilog = 'Author: Akhlak Mahmood, Yingling Group, NCSU')

parser.add_argument('--show', help="Display the plots", action="store_true")
cargs = parser.parse_args()

# %% Load data
data = pd.read_csv("Data/imputed_data.mice.csv")
dobs = data[data.imp == 0]
dimp = data[data.imp > 0]
dimp1 = data[data.imp == 1]
dimp2 = data[data.imp == 2]
dimp3 = data[data.imp == 3]
dimp4 = data[data.imp == 4]
dimp5 = data[data.imp == 5]

dcomplete = dobs.dropna()
# %%
print("Complete observations")
print("Rows", dcomplete.shape[0])

print("Imputed observations")
print("Rows", dimp1.shape[0])

# %% Correlation between the observed features
plotlib.corrplot(
    dobs.drop(columns=["id", "imp", "quality"]),
    method="pearson",
    output="Plots/pearson_corr.observed.png",
)

shifts = features.Differences(dobs).drop(columns=dobs.columns)
plotlib.corrplot(
    shifts,
    method="pearson",
    output="Plots/pearson_corr_shifts.observed.png",
)


plotlib.corrplot(
    dobs.drop(columns=["id", "imp", "quality"]),
    method="spearman",
    output="Plots/spearman_corr.observed.png",
)

# %% Correlation between the imputed features
plotlib.corrplot(
    dimp.drop(columns=["id", "imp", "quality"]),
    method="pearson",
    output="Plots/pearson_corr.imputed.png",
)


shifts = features.Differences(dimp).drop(columns=dimp.columns)
plotlib.corrplot(
    shifts,
    method="pearson",
    output="Plots/pearson_corr_shifts.imputed.png",
)


plotlib.corrplot(
    dimp.drop(columns=["id", "imp", "quality"]),
    method="spearman",
    output="Plots/spearman_corr.imputed.png",
)

# %% Histrogram of the features
def plot_origFeat_histograms(df, output = None):
    colors = ["#333c", "#c00c", "#0ccc"]
    fig, ax = plt.subplots(4, 3, figsize=(6, 5))
    for i in range(3):
        for j, col in enumerate(['tspk', 'lspk', 'tsfw', 'lsfw']):
            colname = "%s%d" %(col, i+1)
            ax[j, i].hist(df[colname], bins=10, color = colors[i])

    for j, col in enumerate(['TSPK', 'LSPK', 'TSFW', 'LSFW']):
        ax[j, 0].set(ylabel = col)
        ax[j, 1].sharex(ax[j, 0])
        ax[j, 2].sharex(ax[j, 0])

    step_names = ["AuNR in H$2$O", "After reaction", "After purification"]
    for i in range(3):
        ax[0, i].set(title = step_names[i])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.20, wspace=0.15)
    if output:
        plt.savefig(output, dpi=600)
        print("Save OK:", output)
    if cargs.show:
        plt.show()
    else:
        plt.close()
# %%
plot_origFeat_histograms(dobs, "Plots/origFeats_dist.observed.png")
plot_origFeat_histograms(dimp, "Plots/origFeats_dist.imputed.png")

# %% Aggregate features
def plot_shifted_histograms(df, output = None):
    colors = ["#c00c", "#0ccc"]
    shifts = ["21", "32"]

    fig, ax = plt.subplots(4, 2, figsize=(5, 5))
    for i in range(2):
        for j, col in enumerate(['tp', 'lp', 'tw', 'lw']):
            colname = "%s%s" %(col, shifts[i])
            ax[j, i].hist(df[colname], bins=10, color = colors[i])

    for j, col in enumerate(['TSPR peak\nshift (nm)',
                             'LSPR peak\nshift (nm)',
                             'TSPR peak\nwidening (nm)',
                             'LSPR peak\nwidening (nm)']):
        ax[j, 0].set(ylabel = col)
        ax[j, 1].sharex(ax[j, 0])

    step_names = ["After reaction", "After purification"]
    for i in range(2):
        ax[0, i].set(title = step_names[i])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.20, wspace=0.15)
    if output:
        plt.savefig(output, dpi=600)
        print("Save OK:", output)
    if cargs.show:
        plt.show()
    else:
        plt.close()

# %%
plot_shifted_histograms(features.Differences(dobs),
                        "Plots/origFeats_shift_dist.observed.png")
plot_shifted_histograms(features.Differences(dimp),
                        "Plots/origFeats_shift_dist.imputed.png")
# %%
