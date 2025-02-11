processSeuratFile <- function(file_name, overwrite = TRUE, delete_h5s = TRUE, download_channel = 63) {
  # Set the download channel environment variable for non-interactive downloads.
  Sys.setenv(SEURATDISK_DOWNLOAD_CHANNEL = as.character(download_channel))
  message(sprintf("Download channel set to %s", download_channel))

  # Install packages if you haven't already
  # run them in the console if they don't have the right to write
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    install.packages("Seurat")
  }
  if (!requireNamespace("SeuratObject", quietly = TRUE)) {
    install.packages("SeuratObject")
  }
  if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes")
    }
    remotes::install_github("mojaveazure/seurat-disk")
  }

  # Load required libraries
  library(Seurat)
  library(SeuratDisk)
  library(SeuratObject)

  # Construct the source file path (assumes RDS is in data/)
  source_file <- sprintf("data/%s.Rds", file_name)
  if (!file.exists(source_file)) {
    stop(sprintf("Source file '%s' does not exist.", source_file))
  }

  # Load your original Seurat object from an RDS file
  my_object <- readRDS(source_file)
  print(my_object)

  # Ensure directories exist for saving output files
  # Directory for h5Seurat files (assumed in data/)
  if (!dir.exists("data/")) {
    dir.create("data/", recursive = TRUE)
    message("Created directory: data/")
  }
  # Directory for h5ad files (assumed in data/processed/)
  if (!dir.exists("data/processed/")) {
    dir.create("data/processed/", recursive = TRUE)
    message("Created directory: data/processed/")
  }

  # Loop over each assay (modality) in the Seurat object
  assay_names <- names(my_object@assays)
  for (assay in assay_names) {
    message(sprintf("Processing assay: %s", assay))

    # Set the current assay as the default so that it becomes the primary (.X) in conversion
    DefaultAssay(my_object) <- assay

    # Construct file prefixes and filenames for this assay.
    # The h5Seurat file is saved in data/ and the h5ad file is intended to be in data/processed/
    prefix_h5s <- sprintf("data/%s_%s", file_name, assay)
    h5s_file <- sprintf("%s.h5Seurat", prefix_h5s)

    prefix_h5ad <- sprintf("data/processed/%s_%s", file_name, assay)
    h5ad_file <- sprintf("%s.h5ad", prefix_h5ad)

    # Save the Seurat object as an h5Seurat file for this assay.
    if (overwrite || !file.exists(h5s_file)) {
      SaveH5Seurat(my_object, filename = h5s_file, overwrite = overwrite)
      message(sprintf("Saved h5Seurat file for assay '%s' at: %s", assay, h5s_file))
    } else {
      message(sprintf("h5Seurat file for assay '%s' exists; using existing file.", assay))
    }

    # Convert the h5Seurat file to an h5ad file for this assay.
    # Note: Convert() writes the h5ad file in the same directory as the h5Seurat file.
    if (overwrite || !file.exists(h5ad_file)) {
      Convert(h5s_file, dest = "h5ad", overwrite = overwrite)
      # Determine the converted file name (it will have the same base as h5s_file, but with a .h5ad extension)
      converted_file <- sub("\\.h5Seurat$", ".h5ad", h5s_file)
      if (file.exists(converted_file)) {
        # Move the file to the processed folder if it's not already there.
        file.rename(converted_file, h5ad_file)
        message(sprintf("Converted and moved h5ad file for assay '%s' to: %s", assay, h5ad_file))
      } else {
        message(sprintf("Expected converted file '%s' not found.", converted_file))
      }
    } else {
      message(sprintf("h5ad file for assay '%s' exists; using existing file.", assay))
    }

    # Delete the h5Seurat file if the option is enabled.
    if (delete_h5s && file.exists(h5s_file)) {
      file.remove(h5s_file)
      message(sprintf("Deleted h5Seurat file for assay '%s' at: %s", assay, h5s_file))
    }
  }
}

# download datasets here first: https://zenodo.org/records/6348128

# Retrieve command line arguments (if any)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  file_name <- "LUNG-CITE"
  message("No file name provided as argument. Using default: LUNG-CITE")
} else {
  file_name <- args[1]
}

# Call the function with the provided (or default) file name.
processSeuratFile(file_name = file_name, overwrite = TRUE, delete_h5s = TRUE, download_channel = 63)