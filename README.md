# Requirements
## Python
- Python 3.0+
- numpy
- pandas
- matplotlib
- scipy

## R
- dplyr
- data.table
- Amelia
- corrplot
- mice

# Usage
`x1_plot_spectra.py` - Plot the UV-absorbance spectra data saved in a training or testing CSV file. Output will be saved to `Plots/` directory.

`x2_extract_features.py` - Generate the training or testing dataset by calculating the peak position and full-width half-max of transverse and longitudinal surface plasmon resonance for training or testing csv.

`x3_data_map.R` - Generate missingness plot and correlation plots for `Data/training.csv`. The CSV file must exist.

