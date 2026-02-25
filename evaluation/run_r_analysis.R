# ================================================================
# R Wrapper (wird von Python aufgerufen)
# ================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

# 1 action: "tables" | "long" | "plots" | "all"
# 2 base_path
# 3 metric: "accuracy" | "f1"
# 4 out_dir
# 5 show_sd: "0" | "1" (optional)

if (length(args) < 4) {
  stop("Usage: Rscript run_r_analysis.R <action> <base_path> <metric> <out_dir> [show_sd]")
}

action    <- args[1]
base_path <- args[2]
metric    <- args[3]
out_dir   <- args[4]
show_sd   <- if (length(args) >= 5) as.logical(as.integer(args[5])) else FALSE

this_dir <- dirname(normalizePath(sys.frame(1)$ofile))
source(file.path(this_dir, "confusion_metrics_functions.R"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_tables <- function(tables, out_dir) {
  if (!length(tables)) {
    fwrite(data.table(), file.path(out_dir, "tables_index.csv"))
    return(invisible(NULL))
  }

  index_dt <- rbindlist(lapply(seq_along(tables), function(i) {
    dt <- tables[[i]]
    data.table(
      i = i,
      dir = dt$dir[1],
      name = dt$name[1],
      unlabeled = dt$unlabeled[1],
      classifier = dt$classifier[1],
      decision = dt$decision[1]
    )
  }), fill = TRUE)

  fwrite(index_dt, file.path(out_dir, "tables_index.csv"))

  for (i in seq_along(tables)) {
    fwrite(tables[[i]], file.path(out_dir, sprintf("table_%04d.csv", i)))
  }

  invisible(NULL)
}

if (action %in% c("tables", "all")) {
  tables <- build_results_tables(base_path, metric)
  write_tables(tables, out_dir)
}

if (action %in% c("long", "plots", "all")) {
  res_long <- build_results_long(base_path, metric)
  fwrite(res_long, file.path(out_dir, "res_long.csv"))
}

if (action %in% c("plots", "all")) {
  res_long <- build_results_long(base_path, metric)

  p1 <- plot_accuracy_curves(res_long, show_sd = show_sd)
  p2 <- plot_mean_accuracy_by_decision(res_long)

  ggsave(file.path(out_dir, "plot_accuracy_curves.pdf"), p1, width = 9, height = 5)
  ggsave(file.path(out_dir, "plot_mean_accuracy_by_decision.pdf"), p2, width = 9, height = 5)
}

cat(sprintf("OK: action=%s base_path=%s metric=%s out_dir=%s\n", action, base_path, metric, out_dir))
