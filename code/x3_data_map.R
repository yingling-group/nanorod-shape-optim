library(dplyr)
library(data.table)
library(Amelia)
library(corrplot)

font <- "CMU Sans Serif"
raw <- fread("Data/all_spectra.csv")
str(raw)


svg("Plots/MissingMap.svg")
missmap(raw, family = font)
dev.off()

## CorrPlot
ignoreCols <- c("name", "teosVolPct")

raw %>% select(-all_of(ignoreCols)) %>% 
  cor(use = "complete.obs", method = "pearson") -> cm

svg("Plots/PearsonCorrelations.svg")
corrplot(cm,
         type="lower", method="number", tl.pos="ld",
         diag = T,  col = COL2('RdYlBu'), tl.col = 1,
         cl.pos="b", number.cex = 0.6, tl.cex = 0.6,
         family = font
        )

dev.off()

raw %>% select(-all_of(ignoreCols)) %>% 
  cor(use = "complete.obs", method = "spearman") -> cm

svg("Plots/SpearmanCorrelations.svg")
corrplot(cm,
         type="lower", method="number", tl.pos="ld",
         diag = T,  col = COL2('RdYlBu'), tl.col = 1,
         cl.pos="b", number.cex = 0.6, tl.cex = 0.6,
         family = font
        )

dev.off()
