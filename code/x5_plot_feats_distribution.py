# %%
import argparse
import numpy as np
import pandas as pd 
from model import plotlib
from model.AdFeatures import Differences

import matplotlib.pyplot as plt
plt.style.use("code/matplotlib.mplstyle")


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

# sort columns
igcols = ['teosVolume', 'teosVolPct', 'full', 'lobe', 'other']
scols = sorted(dobs.columns.drop(igcols))
dobs = dobs[scols+igcols]

# %% Correlation between the observed features
plotlib.corrplot(
    dobs.drop(columns=["id", "imp", "quality"]),
    method="pearson",
    output="Plots/pearson_corr.observed.png",
    figsize = (4, 4), wcbar = False
)

plotlib.corrplot(
    dobs.drop(columns=["id", "imp", "quality"]),
    method="spearman",
    output="Plots/spearman_corr.observed.png",
    figsize = (4, 4), wcbar = False
)


# Drop the original X columns
igcols = ['full', 'lobe', 'other']
toDrop = dobs.columns.difference(igcols)
shifts = Differences(dobs).drop(columns=toDrop)

# sort columns
scols = sorted(shifts.columns.drop(igcols))
shifts = shifts[scols+igcols]

plotlib.corrplot(
    shifts,
    method="pearson",
    output="Plots/pearson_corr_shifts.observed.png",
)

plotlib.corrplot(
    shifts,
    method="spearman",
    output="Plots/spearman_corr_shifts.observed.png",
)


# %% Correlation between the imputed features
# plotlib.corrplot(
#     dimp.drop(columns=["id", "imp", "quality"]),
#     method="pearson",
#     output="Plots/pearson_corr.imputed.png",
# )

# # Drop the original X columns
# toDrop = dimp.columns.difference(["full", "lobe", "other"])
# shifts = Differences(dimp).drop(columns=toDrop)
# plotlib.corrplot(
#     shifts,
#     method="pearson",
#     output="Plots/pearson_corr_shifts.imputed.png",
# )


# plotlib.corrplot(
#     dimp.drop(columns=["id", "imp", "quality"]),
#     method="spearman",
#     output="Plots/spearman_corr.imputed.png",
# )

# # %% Histrogram of the outputs
# def plot_response_histograms(df, output = None):
#     colors = ["#333c", "#c00c", "#0ccc"]
#     fig, ax = plt.subplots(1, 3, figsize=(6, 2.1), sharex=True)
#     for j, colname in enumerate(['full', 'lobe', 'other']):
#         ax[j].hist(df[colname], color = colors[j], density=True, label = colname.capitalize())
#         # ax[j].set(xlabel = colname.capitalize())
#         ax[j].legend(prop={'family': 'cursive'})

#     ax[0].set(ylabel = "Frequency", xticks=np.arange(0, 1.01, 0.25))
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.18)
#     if output:
#         plt.savefig(output)
#         print("Save OK:", output)
#     if cargs.show:
#         plt.show()
#     else:
#         plt.close()
        
# # %%
# plot_response_histograms(dobs, "Plots/respFeats_dist.observed.png")
# # plot_response_histograms(dimp, "Plots/respFeats_dist.imputed.png")


# # %% Histrogram of the features
# def plot_origFeat_histograms(df, output = None):
#     colors = ["#333c", "#c00c", "#0ccc"]
#     fig, ax = plt.subplots(4, 3, figsize=(6, 5))
#     for i in range(3):
#         for j, col in enumerate(['tspk', 'lspk', 'tsfw', 'lsfw']):
#             colname = "%s%d" %(col, i+1)
#             ax[j, i].hist(df[colname], bins=10, color = colors[i])

#     for j, col in enumerate(['TSPK', 'LSPK', 'TSFW', 'LSFW']):
#         ax[j, 0].set(ylabel = col)
#         ax[j, 1].sharex(ax[j, 0])
#         ax[j, 2].sharex(ax[j, 0])

#     step_names = ["AuNR in $H_2 O$", "After reaction", "After purification"]
#     for i in range(3):
#         ax[0, i].set(title = step_names[i])

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.20, wspace=0.15)
#     if output:
#         plt.savefig(output, dpi=600)
#         print("Save OK:", output)
#     if cargs.show:
#         plt.show()
#     else:
#         plt.close()
# # %%
# plot_origFeat_histograms(dobs, "Plots/origFeats_dist.observed.png")
# plot_origFeat_histograms(dimp, "Plots/origFeats_dist.imputed.png")

# # %% Aggregate features
# def plot_shifted_histograms(df, output = None):
#     colors = ["#c00c", "#0ccc"]
#     shifts = ["21", "32"]

#     fig, ax = plt.subplots(4, 2, figsize=(5, 5))
#     for i in range(2):
#         for j, col in enumerate(['tp', 'lp', 'tw', 'lw']):
#             colname = "%s%s" %(col, shifts[i])
#             ax[j, i].hist(df[colname], bins=10, color = colors[i])

#     for j, col in enumerate(['TSPR peak\nshift (nm)',
#                              'LSPR peak\nshift (nm)',
#                              'TSPR peak\nwidening (nm)',
#                              'LSPR peak\nwidening (nm)']):
#         ax[j, 0].set(ylabel = col)
#         ax[j, 1].sharex(ax[j, 0])

#     step_names = ["After reaction", "After purification"]
#     for i in range(2):
#         ax[0, i].set(title = step_names[i])

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.20, wspace=0.15)
#     if output:
#         plt.savefig(output, dpi=600)
#         print("Save OK:", output)
#     if cargs.show:
#         plt.show()
#     else:
#         plt.close()

# # %%
# plot_shifted_histograms(Differences(dobs),
#                         "Plots/origFeats_shift_dist.observed.png")
# plot_shifted_histograms(Differences(dimp),
#                         "Plots/origFeats_shift_dist.imputed.png")
# # %%
