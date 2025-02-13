file_name <- "LUNG-CITE"

source_file <- sprintf("/Users/dricpro/PycharmProjects/MultiOmicsIntegration/datasets/data/%s.Rds", file_name)
if (!file.exists(source_file)) {
  stop(sprintf("Source file '%s' does not exist.", source_file))
}

my_object <- readRDS(source_file)
DefaultAssay(my_object) <- "RNA"
print(my_object)

length(unique(colnames(my_object)))