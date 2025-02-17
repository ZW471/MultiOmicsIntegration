processSeuratFile <- function(file_name, overwrite = TRUE, delete_temp = TRUE, download_channel = 63) {
  # Set the download channel environment variable for non-interactive downloads.
  Sys.setenv(SEURATDISK_DOWNLOAD_CHANNEL = as.character(download_channel))
  message(sprintf("Download channel set to %s", download_channel))

  # Install and load required packages
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
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    install.packages("hdf5r")
  }

  library(Seurat)
  library(SeuratDisk)
  library(SeuratObject)
  library(hdf5r)

  # Construct the source file path (assumes RDS is in data/)
  source_file <- sprintf("data/%s.Rds", file_name)
  if (!file.exists(source_file)) {
    stop(sprintf("Source file '%s' does not exist.", source_file))
  }

  # Load the original Seurat object from an RDS file
  my_object <- readRDS(source_file)
  print(my_object)

  # Ensure output directories exist
  if (!dir.exists("data/")) {
    dir.create("data/", recursive = TRUE)
    message("Created directory: data/")
  }
  if (!dir.exists("data/processed/")) {
    dir.create("data/processed/", recursive = TRUE)
    message("Created directory: data/processed/")
  }

  # Create a list to record the assay names and corresponding h5ad file paths
  assay_h5ad_list <- list()

  # Loop over each assay (modality) in the Seurat object
  assay_names <- names(my_object@assays)
  for (assay in assay_names) {
    message(sprintf("Processing assay: %s", assay))

    # Set the current assay as the default
    DefaultAssay(my_object) <- assay

    # Construct file prefixes for this assay
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
    if (overwrite || !file.exists(h5ad_file)) {
      Convert(h5s_file, dest = "h5ad", overwrite = overwrite)
      # The converted file will have the same base name as h5s_file, but with .h5ad extension.
      converted_file <- sub("\\.h5Seurat$", ".h5ad", h5s_file)
      if (file.exists(converted_file)) {
        # Move the converted file to the processed folder (if necessary)
        file.rename(converted_file, h5ad_file)
        message(sprintf("Converted and moved h5ad file for assay '%s' to: %s", assay, h5ad_file))
      } else {
        message(sprintf("Expected converted file '%s' not found.", converted_file))
      }
    } else {
      message(sprintf("h5ad file for assay '%s' exists; using existing file.", assay))
    }

    # Record the assay and its h5ad file path.
    assay_h5ad_list[[assay]] <- h5ad_file

    # Delete the temporary h5Seurat file if requested.
    if (delete_temp && file.exists(h5s_file)) {
      file.remove(h5s_file)
      message(sprintf("Deleted temporary h5Seurat file for assay '%s' at: %s", assay, h5s_file))
    }
  }

  # ---- Combine the per-assay h5ad files into one .h5 file ------------------
  combined_h5_path <- sprintf("data/processed/%s.h5", file_name)
  message(sprintf("Creating combined HDF5 file at: %s", combined_h5_path))

  # Create (or overwrite) the combined HDF5 file and a root group "mod"
  combined_file <- H5File$new(combined_h5_path, mode = "w")
  mod_group <- combined_file$create_group("mod")

  # Define a recursive function to copy the contents of one HDF5 group into another
  copy_h5_recursive <- function(src_group, dest_group) {
    # List immediate children in the source group
    items <- src_group$ls(recursive = FALSE)
    if (nrow(items) == 0) return()
    for (i in seq_len(nrow(items))) {
      child_name <- items$name[i]
      otype_val <- items$otype[i]
      # If otype is missing, skip this item
      if (length(otype_val) == 0) {
        next
      }
      # Open the child object (it may be a group or dataset)
      child_obj <- src_group[[child_name]]
      if (otype_val == "H5I_GROUP") {
        # Create a new group in the destination and copy recursively
        new_group <- dest_group$create_group(child_name)
        copy_h5_recursive(child_obj, new_group)
      } else if (otype_val == "H5I_DATASET") {
        # Read the dataset and write it into the destination
        data <- child_obj[]
        dest_group$create_dataset(child_name, data = data)
      }
      # (Optionally, attributes could also be copied here.)
    }
  }

  # For each assay, open its h5ad file and copy its contents under mod/<assay>
  for (assay in names(assay_h5ad_list)) {
    h5ad_file <- assay_h5ad_list[[assay]]
    message(sprintf("Adding assay '%s' to combined file from: %s", assay, h5ad_file))
    if (!file.exists(h5ad_file)) {
      warning(sprintf("h5ad file for assay '%s' not found. Skipping.", assay))
      next
    }
    h5ad_obj <- H5File$new(h5ad_file, mode = "r")
    # Create a subgroup for this assay under the "mod" group.
    assay_group <- mod_group$create_group(assay)
    # Copy all contents from the root of the h5ad file into the assay group.
    copy_h5_recursive(h5ad_obj, assay_group)
    h5ad_obj$close_all()

    # Optionally, delete the individual h5ad file after merging.
    if (delete_temp) {
      file.remove(h5ad_file)
      message(sprintf("Deleted temporary h5ad file for assay '%s' at: %s", assay, h5ad_file))
    }
  }

  # Close the combined file.
  combined_file$close_all()
  message(sprintf("Combined HDF5 file created successfully: %s", combined_h5_path))
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
processSeuratFile(file_name = file_name, overwrite = TRUE, delete_temp = TRUE, download_channel = 63)