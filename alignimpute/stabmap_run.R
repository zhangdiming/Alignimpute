suppressPackageStartupMessages({library(StabMap); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE); dir <- args[1]; refarg <- args[2]
meta <- fromJSON(file.path(dir, "meta.json"))
assay_list <- list()
for (i in seq_len(nrow(meta))) {
  tag <- meta$dataset[i]
  X <- as.matrix(read.csv(file.path(dir, paste0(tag, ".csv")), header = FALSE))
  rownames(X) <- readLines(file.path(dir, paste0(tag, ".features.txt")))
  colnames(X) <- readLines(file.path(dir, paste0(tag, ".cells.txt")))
  assay_list[[tag]] <- X
}
refs <- if (refarg == "all") meta$dataset[meta$reference] else strsplit(refarg, ",")[[1]]
print(mosaicDataTopology(assay_list))
set.seed(0)
scl <- if (length(args) >= 3 && args[3] == "noscale") FALSE else TRUE
emb <- stabMap(assay_list, reference_list = refs, ncomponentsReference = 50, ncomponentsSubset = 50,
               projectAll = TRUE, maxFeatures = 5000, plot = FALSE, scale.center = TRUE, scale.scale = scl)
if (length(refs) > 1) emb <- reWeightEmbedding(emb)
tagout <- paste0(if (refarg == "all") "all" else paste0("ref_", gsub(",", "_", refarg)), if (scl) "" else "_noscale")
write.csv(emb, file.path(dir, paste0("embedding_", tagout, ".csv")))
cat("embedding", dim(emb), "\n")
