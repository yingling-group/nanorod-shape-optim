library(dplyr)
library(data.table)
library(mice)
library(ggplot2)

raw <- fread("Data/training.csv")
str(raw)

outdir <- "Data"
dir.create(outdir, showWarnings = FALSE)

## MICE
## =============================================================
m  = 5       # number of imputations

# Init the mice code with 0 iterations 
imp <- mice(raw, maxit = 0)

# Extract predictorMatrix and methods of imputation
predM <- imp$predictorMatrix
meth <- imp$method

# Do not use the following variables as predictors
predM[, c("name", "quality1", "quality2", "thickness")] <- 0

## Run
rds <- paste0(outdir, "/rds.lastrun_mice.rds")

# reverse the ! to rerun.
if (!file.exists(rds)) {
  print("Running mice ...")
  t1 <- Sys.time()
  
  # Use predM as the predictor matrix.
  impObj <- mice(
    raw,
    maxit = m,
    predictorMatrix = predM,
    method = meth,
    print =  TRUE
  )
  
  dt <- Sys.time() - t1
  print(dt)
  
  print("Done!")
  
  lastrun <- list(impObj = impObj, dt = dt)
  save(lastrun, file = rds)
  
} else {
  # load last run
  load(file = rds)
  impObj <- lastrun$impObj
  dt <- lastrun$dt
}
gc()
impObj

imputed <- mice::complete(impObj, action="long", include=T)
imputed <- imputed %>% rename("id" = ".id", "imp" = ".imp")

fwrite(imputed, "Data/imputed_data.mice.csv")

## Calculate mean of the imputed data
## =============================================================
imputed %>% 
  filter(imp > 0) %>% 
  group_by(id) %>% 
  summarise(across(everything(), list(mean))) -> impmean

colnames(impmean) <- colnames(imputed)
fwrite(impmean, "Data/imputed_data_mean.mice.csv")

## Calculate std. dev. of the imputed data
## =============================================================
imputed %>% 
  filter(imp > 0) %>% 
  group_by(id) %>% 
  summarise(across(everything(), list(sd))) -> impstd

colnames(impstd) <- colnames(imputed)
fwrite(impstd, "Data/imputed_data_std.mice.csv")

## Perform KS test on the imputed data
## =============================================================
m <- 5
kstable <- data.table()

for (col in colnames(imputed)) {
  if (col %in% c('id', 'imp', "name", "quality1", "quality2", "thickness")) {
    next
  }
  cat("Processing ", col, "\n")
  ksdist <- matrix(nrow = m, ncol = m)
  kspval <- matrix(nrow = m, ncol = m)

  for (i in seq(m)) {
    for (j in seq(m)) {
      if (i == j) {
        ksdist[i, j] <- 0
        kspval[i, j] <- 1
        next
      }

      imp1 <- imputed[imputed$imp==i,][[col]]
      imp2 <- imputed[imputed$imp==j,][[col]]
      
      k <- suppressWarnings(ks.test(imp1, imp2))
      ksdist[i, j] <- k$statistic
      kspval[i, j] <- k$p.value
    }
    
  }
  
  res <- list(
    ks.stat = ksdist,
    ks.pval = kspval,
    mean.dist = mean(ksdist),
    mean.pval = mean(kspval),
    dist.sd = sd(ksdist),
    pval.sd = sd(kspval)
  )
  
  kstable <- rbind(
    kstable,
    data.frame(
      param = col,
      mean.ks.dist = res$mean.dist,
      mean.ks.pval = res$mean.pval,
      sd.ks.dist = res$dist.sd,
      sd.ks.pval = res$pval.sd
    )
  )
}

kstable
fwrite(kstable, "Data/imputed_ks_test.mice.csv")

## Plot KS test result
dir.create("Plots", showWarnings = FALSE)

kstable %>% 
    select(-mean.ks.pval) %>% 
    ggplot(aes(y=param, x=mean.ks.dist, fill=param)) +
        geom_bar(stat="identity", col='black') +
        # geom_errorbar(aes(xmin=mean.ks.dist-sd.ks.dist, xmax=mean.ks.dist+sd.ks.dist),
        #               width=.2, position=position_dodge(.9)) +
        geom_vline(xintercept = 0.04301, lty=2, col='red') +
        xlim(0, 0.15) +
        theme_bw() + ylab("") + xlab("Kolmogorov-Smirnov test statistic") +
        theme(legend.position = "none")

ggsave("Plots/ks.dist_barplots.mice.svg")

print("All Done!")

