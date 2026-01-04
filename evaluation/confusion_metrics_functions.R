# ================================================================
# Confusion-Matrix Utilities & Aggregation
# ================================================================
# NUR Funktionen: keine View()/print(), keine Ausführung
# ================================================================

library(data.table)
library(ggplot2)

confusion_matrix_from_long <- function(df_idx) {
  true_labels <- sort(unique(df_idx$true))
  pred_labels <- sort(unique(df_idx$pred))
  labels <- sort(union(true_labels, pred_labels))

  label_to_idx <- setNames(seq_along(labels), labels)
  n <- length(labels)

  cm <- matrix(0, n, n, dimnames = list(labels, labels))

  for (i in seq_len(nrow(df_idx))) {
    t <- label_to_idx[[as.character(df_idx$true[i])]]
    p <- label_to_idx[[as.character(df_idx$pred[i])]]
    cm[t, p] <- cm[t, p] + df_idx$count[i]
  }

  list(cm = cm, labels = labels)
}

accuracy_from_cm <- function(cm) {
  total <- sum(cm)
  if (total == 0) return(NaN)
  sum(diag(cm)) / total
}

f1_from_cm <- function(cm) {
  n <- nrow(cm)

  if (n == 2) {
    tp <- cm[2, 2]
    fp <- cm[1, 2]
    fn <- cm[2, 1]
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec  <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    if (prec + rec == 0) return(0)
    return(2 * prec * rec / (prec + rec))
  }

  f1s <- numeric(n)
  for (k in seq_len(n)) {
    tp <- cm[k, k]
    fp <- sum(cm[, k]) - tp
    fn <- sum(cm[k, ]) - tp
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec  <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1s[k] <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
  }
  mean(f1s)
}

aggregate_confusion_metrics <- function(results_dir, metric = "accuracy") {
  stopifnot(metric %in% c("accuracy", "f1"))
  files <- list.files(results_dir, "\\.csv$", full.names = TRUE)
  if (!length(files)) return(NULL)

  values_per_index <- list()

  for (path in files) {
    df <- fread(path)
    required <- c("cm_index", "true", "pred", "count")
    if (!all(required %in% names(df))) next

    for (cm_idx in unique(df$cm_index)) {
      cm <- confusion_matrix_from_long(df[cm_index == cm_idx])$cm
      val <- if (metric == "accuracy") accuracy_from_cm(cm) else f1_from_cm(cm)
      values_per_index[[as.character(cm_idx)]] <-
        c(values_per_index[[as.character(cm_idx)]], val)
    }
  }

  rbindlist(lapply(names(values_per_index), function(idx) {
    vals <- values_per_index[[idx]]
    data.table(
      cm_index = as.integer(idx),
      mean     = mean(vals, na.rm = TRUE),
      sd       = if (length(vals) > 1) sd(vals) else 0,
      n_files  = length(vals)
    )
  }))[order(cm_index)]
}

extract_experiment_name <- function(path, base_path) {
  rel <- sub(paste0("^", normalizePath(base_path), "/?"),
             "", normalizePath(path))
  sub(".*(unlabeled_[^/]+/.*)$", "\\1", rel)
}

parse_experiment_factors <- function(name) {
  parts <- tstrsplit(name, "/", fixed = TRUE)
  list(
    unlabeled  = sub("^unlabeled_",  "", parts[[1]]),
    classifier = sub("^classifier_", "", parts[[2]]),
    decision   = sub("^decision_",   "", parts[[3]])
  )
}

build_results_tables <- function(base_path, metric = "accuracy") {
  csv_files <- list.files(base_path, "\\.csv$", recursive = TRUE, full.names = TRUE)
  dirs_with_csv <- unique(dirname(csv_files))

  res_list <- lapply(dirs_with_csv, function(d) {
    dt <- aggregate_confusion_metrics(d, metric)
    if (is.null(dt)) return(NULL)

    name <- extract_experiment_name(d, base_path)
    f    <- parse_experiment_factors(name)

    dt[, `:=`(
      dir        = d,
      name       = name,
      unlabeled  = f$unlabeled,
      classifier = f$classifier,
      decision   = f$decision
    )]
    dt
  })

  Filter(Negate(is.null), res_list)
}

build_results_long <- function(base_path, metric = "accuracy") {
  tables <- build_results_tables(base_path, metric)
  if (!length(tables)) return(data.table())

  rbindlist(lapply(tables, function(dt) {
    copy(dt)[, `:=`(
      mean_accuracy = mean,
      std_accuracy  = sd
    )]
  }), fill = TRUE)
}

plot_accuracy_curves <- function(res_long, show_sd = FALSE) {
  p <- ggplot(
    res_long,
    aes(
      x = cm_index,
      y = mean_accuracy,
      group = interaction(classifier, decision, unlabeled),
      colour = classifier,
      linetype = decision
    )
  ) +
    geom_line(linewidth = 1) +
    theme_minimal()

  if (show_sd) {
    p <- p +
      geom_ribbon(
        aes(
          ymin = mean_accuracy - std_accuracy,
          ymax = mean_accuracy + std_accuracy,
          fill = classifier
        ),
        alpha = 0.2,
        colour = NA
      ) +
      guides(fill = "none")
  }
  p
}

plot_mean_accuracy_by_decision <- function(res_long) {
  ave_res <- res_long[
    , .(mean = mean(mean_accuracy)),
    by = .(cm_index, unlabeled, decision)
  ]

  ggplot(
    ave_res,
    aes(
      x = cm_index,
      y = mean,
      group  = interaction(decision, unlabeled),
      colour = decision
    )
  ) +
    geom_line(linewidth = 1) +
    theme_minimal() +
    theme(legend.position = "bottom")
}
