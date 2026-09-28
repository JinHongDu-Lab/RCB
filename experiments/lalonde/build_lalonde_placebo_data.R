#!/usr/bin/env Rscript

# Build leakage-safe LaLonde placebo datasets.  The hidden placebo outcome is
# excluded from the feature map, scaling fit, and common-support construction.

suppressPackageStartupMessages(library(foreign))
suppressPackageStartupMessages(library(nnet))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
here <- dirname(script_path)
data_dir <- file.path(here, "..", "..", "data", "lalonde")

dw <- read.dta(file.path(data_dir, "nsw_dw.dta"), convert.factors = FALSE)
psid <- read.dta(
  file.path(data_dir, "psid_controls.dta"), convert.factors = FALSE
)
df <- rbind(dw[dw$treat == 1, ], psid)

demographic_features <- function(d) {
  squares <- cbind(d$age^2, d$education^2)
  interactions <- model.matrix(
    ~ (age + education + married + black + hispanic + nodegree)^2 - 1,
    data = d
  )
  cbind(interactions, squares)
}

re75_features <- function(d) {
  base <- demographic_features(d)
  earnings <- cbind(d$re74, d$re74^2, d$re74 == 0)
  interactions <- cbind(
    d$re74 * d$age,
    d$re74 * d$education,
    d$re74 * d$married,
    d$re74 * d$black,
    d$re74 * d$hispanic,
    d$re74 * d$nodegree
  )
  cbind(base, earnings, interactions)
}

build_one <- function(outcome, raw_features, label) {
  scaled <- scale(raw_features)
  keep_columns <- apply(scaled, 2, function(z) all(is.finite(z)))
  X <- scaled[, keep_columns, drop = FALSE]
  T <- df$treat
  fit <- multinom(T ~ X - 1, trace = FALSE)
  propensity <- fit$fitted.values
  treated_propensity <- propensity[T == 1]
  keep <- propensity >= min(treated_propensity) &
    propensity <= max(treated_propensity)
  out <- data.frame(y = outcome[keep], treat = T[keep], X[keep, ])
  names(out) <- c(
    "y", "treat", sprintf("x%03d", seq_len(ncol(out) - 2))
  )
  path <- file.path(data_dir, sprintf("lalonde_placebo_%s.csv", label))
  write.csv(out, path, row.names = FALSE)
  cat(sprintf(
    "Wrote %s: p=%d, treated=%d, controls=%d\n",
    path, ncol(out) - 2, sum(out$treat == 1), sum(out$treat == 0)
  ))
}

# For the 1974 placebo, exclude both re74 and the later re75 measurement.
build_one(df$re74, demographic_features(df), "re74")

# For the 1975 placebo, re74 and its pre-specified transformations are allowed.
build_one(df$re75, re75_features(df), "re75")
