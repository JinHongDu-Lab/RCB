#!/usr/bin/env Rscript

# Construct the 171-feature NSW randomized-control target with exactly the
# feature map and scaling used by the official LaLonde-171 analysis matrix.
# The existing training CSV is treated as an immutable reference: this script
# reconstructs it and stops unless every retained feature agrees numerically.

suppressPackageStartupMessages(library(foreign))
suppressPackageStartupMessages(library(nnet))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
here <- dirname(script_path)
data_dir <- file.path(here, "..", "..", "data", "lalonde")
reference_path <- file.path(data_dir, "lalonde171_psid_common_support.csv")
output_path <- file.path(data_dir, "lalonde171_nsw_control_target.csv")

dw <- read.dta(file.path(data_dir, "nsw_dw.dta"), convert.factors = FALSE)
psid_control <- read.dta(
  file.path(data_dir, "psid_controls.dta"), convert.factors = FALSE
)
treated <- dw[dw$treat == 1, ]
nsw_control <- dw[dw$treat == 0, ]
train_df <- rbind(treated, psid_control)

raw_specs <- function(df, poly_fit = NULL) {
  X1 <- cbind(
    df$age, df$education, df$married, df$black, df$hispanic,
    df$re74, df$re75, df$age^2, df$education^2, df$re74^2, df$re75^2
  )
  X2 <- cbind(X1, df$re74 == 0, df$re75 == 0, df$nodegree)
  Xmain <- cbind(
    df$age, df$education, df$married, df$black, df$hispanic,
    df$re74, df$re75, df$nodegree, df$re74 == 0, df$re75 == 0
  )
  Xcont <- cbind(
    df$age * df$married, df$age * df$nodegree,
    df$age * df$black, df$age * df$hispanic,
    df$age * (df$re74 == 0), df$age * (df$re75 == 0),
    df$education * df$married, df$education * df$nodegree,
    df$education * df$black, df$education * df$hispanic,
    df$education * (df$re74 == 0), df$education * (df$re75 == 0),
    df$re74 * df$married, df$re74 * df$nodegree,
    df$re74 * df$black, df$re74 * df$hispanic,
    df$re74 * (df$re75 == 0),
    df$re75 * df$married, df$re75 * df$nodegree,
    df$re75 * df$black, df$re75 * df$hispanic,
    df$re75 * (df$re74 == 0)
  )
  Xdummy <- cbind(
    df$married * df$nodegree, df$married * df$black,
    df$married * df$hispanic, df$married * (df$re74 == 0),
    df$married * (df$re75 == 0), df$nodegree * df$black,
    df$nodegree * df$hispanic, df$nodegree * (df$re74 == 0),
    df$nodegree * (df$re75 == 0), df$black * (df$re74 == 0),
    df$black * (df$re75 == 0), df$hispanic * (df$re74 == 0),
    df$hispanic * (df$re75 == 0),
    (df$re74 == 0) * (df$re75 == 0)
  )
  poly_input <- cbind(df$age, df$education, df$re74, df$re75)
  Xpoly <- if (is.null(poly_fit)) {
    poly(poly_input, degree = 5)
  } else {
    predict(poly_fit, newdata = poly_input)
  }
  X3 <- cbind(Xmain, Xcont, Xdummy, Xpoly)
  stopifnot(ncol(X1) == 11, ncol(X2) == 14, ncol(X3) == 171)
  list(X1 = X1, X2 = X2, X3 = X3, poly_fit = Xpoly)
}

scale_fit <- function(A) {
  scaled <- scale(A)
  list(
    values = scaled,
    center = attr(scaled, "scaled:center"),
    scale = attr(scaled, "scaled:scale")
  )
}

scale_apply <- function(A, fit) {
  sweep(sweep(A, 2, fit$center, "-"), 2, fit$scale, "/")
}

train_raw <- raw_specs(train_df)
target_raw <- raw_specs(nsw_control, poly_fit = train_raw$poly_fit)
fits <- lapply(train_raw[c("X1", "X2", "X3")], scale_fit)
X_train <- lapply(fits, function(x) x$values)
X_target <- Map(
  function(name, fit) scale_apply(target_raw[[name]], fit),
  c("X1", "X2", "X3"), fits
)

T <- train_df$treat
support <- function(A) {
  p <- multinom(T ~ A - 1, trace = FALSE)$fitted.values
  p >= min(p[T == 1]) & p <= max(p[T == 1])
}
keep <- support(X_train[[1]]) & support(X_train[[2]]) & support(X_train[[3]])

reconstructed <- data.frame(
  y = train_df$re78[keep], treat = T[keep], X_train[[3]][keep, ]
)
names(reconstructed) <- c(
  "y", "treat", sprintf("x%03d", seq_len(ncol(reconstructed) - 2))
)
reference <- read.csv(reference_path, check.names = FALSE)
stopifnot(identical(dim(reconstructed), dim(reference)))
numeric_gap <- max(abs(as.matrix(reconstructed) - as.matrix(reference)))
if (!is.finite(numeric_gap) || numeric_gap > 1e-8) {
  stop(sprintf("Reconstruction disagrees with reference; max gap %.3e", numeric_gap))
}

target <- data.frame(y = nsw_control$re78, X_target[[3]])
names(target) <- c("y", sprintf("x%03d", seq_len(ncol(target) - 1)))
stopifnot(nrow(target) == 260, ncol(target) == 172)
write.csv(target, output_path, row.names = FALSE)
for (j in seq_along(X_target)) {
  target_j <- data.frame(y = nsw_control$re78, X_target[[j]])
  names(target_j) <- c(
    "y", sprintf("x%03d", seq_len(ncol(target_j) - 1))
  )
  representation_path <- file.path(
    data_dir,
    sprintf("lalonde_nsw_control_target_p%d.csv", ncol(X_target[[j]]))
  )
  write.csv(target_j, representation_path, row.names = FALSE)
}
cat(sprintf(
  "Wrote %s: %d randomized NSW controls, 171 features; reference gap %.3e\n",
  output_path, nrow(target), numeric_gap
))
