#!/usr/bin/env Rscript

# Build the exact 171-feature PSID analysis matrix used by the Auto-DML
# replication package and inherited by Bruns-Smith et al. (2024/2025).
#
# The construction follows the official ECTA18515 replication code:
#   * specifications_intersection.R::get_data_intersection;
#   * stats::poly(..., degree = 5) for the 125-dimensional multivariate
#     orthogonal-polynomial block;
#   * nnet::multinom common-support fits for the 11-, 14-, and 171-feature
#     specifications, retaining the intersection of the three support sets.

suppressPackageStartupMessages(library(foreign))
suppressPackageStartupMessages(library(nnet))

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
here <- dirname(script_path)
data_dir <- file.path(here, "..", "..", "data", "lalonde")
output <- if (length(args) >= 1) args[[1]] else
  file.path(data_dir, "lalonde171_psid_common_support.csv")

dw <- read.dta(file.path(data_dir, "nsw_dw.dta"), convert.factors = FALSE)
psid_control <- read.dta(
  file.path(data_dir, "psid_controls.dta"), convert.factors = FALSE
)
treated <- dw[dw$treat == 1, ]
df <- rbind(treated, psid_control)

make_specs <- function(df) {
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
  Xpoly <- poly(cbind(df$age, df$education, df$re74, df$re75), degree = 5)
  X3 <- cbind(Xmain, Xcont, Xdummy, Xpoly)
  stopifnot(ncol(X1) == 11, ncol(X2) == 14, ncol(X3) == 171)
  list(scale(X1), scale(X2), scale(X3))
}

X <- make_specs(df)
T <- df$treat
support <- function(A) {
  p <- multinom(T ~ A - 1, trace = FALSE)$fitted.values
  p >= min(p[T == 1]) & p <= max(p[T == 1])
}
keep <- support(X[[1]]) & support(X[[2]]) & support(X[[3]])
out <- data.frame(y = df$re78[keep], treat = T[keep], X[[3]][keep, ])
names(out) <- c("y", "treat", sprintf("x%03d", seq_len(ncol(out) - 2)))

stopifnot(
  nrow(out) == 912, ncol(out) == 173,
  sum(out$treat == 1) == 185, sum(out$treat == 0) == 727
)
write.csv(out, output, row.names = FALSE)
for (j in seq_along(X)) {
  out_j <- data.frame(y = df$re78[keep], treat = T[keep], X[[j]][keep, ])
  names(out_j) <- c(
    "y", "treat", sprintf("x%03d", seq_len(ncol(out_j) - 2))
  )
  representation_path <- file.path(
    data_dir, sprintf("lalonde_common_support_p%d.csv", ncol(X[[j]]))
  )
  write.csv(out_j, representation_path, row.names = FALSE)
}
cat(sprintf("Wrote %s: %d rows, %d features, %d treated, %d controls\n",
            output, nrow(out), ncol(out) - 2,
            sum(out$treat == 1), sum(out$treat == 0)))
