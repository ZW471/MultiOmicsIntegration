import h5py
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

SC_TYPES = ["RNA", "ADT", "ATAC"]


def preprocess_scRNA(h5ad_in, h5ad_out=None, n_top_genes=300, n_pcs=100):
    """
    Preprocessing pipeline for single-cell RNA-seq data.
    Handles edge cases like zero variances, NaNs, and ensures valid input for downstream steps.
    """
    adata = sc.read_h5ad(h5ad_in)

    # Convert to dense matrix if sparse
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # 1. Filter cells and genes
    # sc.pp.filter_cells(adata, min_counts=1)  # Remove cells with zero counts
    # sc.pp.filter_genes(adata, min_cells=10)  # Filter low-expressed genes

    # 2. Normalize and handle potential NaNs/negatives
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.X = np.clip(adata.X, 0, None)  # Ensure non-negative values
    adata.X = np.nan_to_num(adata.X)  # Replace any NaNs

    # 3. Log-transform safely
    sc.pp.log1p(adata)
    adata.X = np.nan_to_num(adata.X)  # Post-log cleanup

    # 4. Check if HVG selection is feasible
    try:
        # Compute gene means and variances
        mean = np.mean(adata.X, axis=0)
        variance = np.var(adata.X, axis=0)

        if np.all(variance == 0) or np.any(np.isnan(mean)) or np.any(np.isnan(variance)):
            raise ValueError("Insufficient variance or NaNs detected.")

        # Select HVGs with Seurat method
        sc.pp.highly_variable_genes(
            adata,
            flavor='seurat',
            n_top_genes=n_top_genes,
            min_mean=0.1,
            max_mean=3,
            min_disp=0.5
        )
        hvgs = adata.var[adata.var.highly_variable].index
    except Exception as e:
        print(f"HVG selection failed: {str(e)}. Using all genes.")
        hvgs = adata.var.index

    # 5. Subset to HVGs and proceed with scaling/PCA
    adata = adata[:, hvgs]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='randomized', n_comps=n_pcs)

    if h5ad_out is not None:
        adata.write_h5ad(h5ad_out)

    return adata


def preprocess_ADT(h5ad_in, h5ad_out=None, cofactor=5, min_proteins=8, min_cells=10, n_pcs=100):
    """
    ADT preprocessing pipeline:
      1. Filter low-quality cells & proteins
      2. Apply arcsinh transformation
      3. Scale the data
      4. Perform PCA
    Saves the processed data if h5ad_out is provided.
    """
    adata = sc.read_h5ad(h5ad_in)

    # 1. Filter low-quality cells (based on number of detected proteins)
    # sc.pp.filter_cells(adata, min_counts=min_proteins)
    # sc.pp.filter_genes(adata, min_cells=min_cells)

    # 2. Convert sparse matrix to dense
    if issparse(adata.X):
        adata.X = adata.X.toarray()

    # 3. Apply arcsinh transformation
    adata.X = np.arcsinh(adata.X / cofactor)

    # 4. Scale & PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='randomized', n_comps=n_pcs)

    # Save
    if h5ad_out is not None:
        adata.write_h5ad(h5ad_out)

    return adata


def remove_lsi_key(h5ad_path):
    """
    Opens the .h5ad file with h5py and removes the varm/LSI key
    if it exists. This prevents shape mismatch errors upon reading.
    """
    with h5py.File(h5ad_path, "r+") as f:
        # Navigate to 'varm' group if it exists
        if "varm" in f:
            varm_group = f["varm"]
            # Check if LSI is in varm
            if "LSI" in varm_group:
                print(f"Removing varm/LSI from {h5ad_path} ...")
                del varm_group["LSI"]
            if "LSI.RAW" in varm_group: del varm_group["LSI.RAW"]
            if "LSI_SVD" in varm_group: del varm_group["LSI_SVD"]


def preprocess_scATAC(h5ad_in, h5ad_out=None,
                      binarize=True,
                      min_peaks_per_cell=500,
                      min_cells_per_peak=10,
                      n_pcs=100):
    """
    Minimal scATAC preprocessing:
      1) Remove LSI key if shape-mismatch
      2) Filter low-quality cells & peaks
      3) Binarize
      4) Normalize & log-transform
      5) Scale & PCA
    """
    # 1. Remove LSI key from h5ad if it exists
    remove_lsi_key(h5ad_in)

    # 2. Load data
    adata = sc.read_h5ad(h5ad_in)

    # Filter cells & peaks
    # sc.pp.filter_cells(adata, min_genes=min_peaks_per_cell)
    # sc.pp.filter_genes(adata, min_cells=min_cells_per_peak)

    # Binarize
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    if binarize:
        X = (X > 0).astype(np.float32)
    adata.X = X

    # If nothing remains, skip normalization & PCA
    if adata.n_obs < 2 or adata.n_vars < 2:
        print(f"Warning: After filtering, shape={adata.shape}. Skipping PCA.")
        if h5ad_out is not None:
            adata.write_h5ad(h5ad_out)
        return adata

    # Normalize & log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Scale & PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='randomized', n_comps=n_pcs)

    # Save
    if h5ad_out is not None:
        adata.write_h5ad(h5ad_out)

    return adata

def preprocess_Peaks(h5ad_in, h5ad_out=None, binarize=True, min_peaks_per_cell=500, min_cells_per_peak=10, n_pcs=100):
    """
    Preprocessing pipeline for Peak (or Peaks) modality data.
    Steps:
      1. Load data from the .h5ad file.
      2. Convert to a dense matrix if needed.
      3. Binarize the matrix (presence/absence of peaks).
      4. Optionally filter low-quality cells/peaks (not implemented here but can be added).
      5. Normalize, log-transform, scale, and compute PCA.
    """
    adata = sc.read_h5ad(h5ad_in)
    
    # Convert sparse matrix to dense 
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    # Binarize the data: 1 if peak is detected, 0 otherwise.
    if binarize:
        X = (X > 0).astype(np.float32)
    adata.X = X

    # If there are too few cells or peaks, skip further processing.
    if adata.n_obs < 2 or adata.n_vars < 2:
        print(f"Warning: After filtering, shape={adata.shape}. Skipping PCA.")
        if h5ad_out is not None:
            adata.write_h5ad(h5ad_out)
        return adata

    # Normalize total counts, log-transform, scale, and PCA.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='randomized', n_comps=n_pcs)

    if h5ad_out is not None:
        adata.write_h5ad(h5ad_out)
    return adata

def preprocess(sc_type, h5ad_in, h5ad_out=None, ignore_error=False, **kwargs):
    sc_type = sc_type.upper()
    if sc_type not in SC_TYPES:
        if ignore_error:
            adata = sc.read_h5ad(h5ad_in)
            if h5ad_out is not None:
                adata.write_h5ad(h5ad_out)
            return adata
        else:
            raise ValueError(f"Unsupported sc_type: {sc_type}")

    if sc_type == "RNA":
        return preprocess_scRNA(h5ad_in, h5ad_out, **kwargs)
    elif sc_type == "ADT":
        return preprocess_ADT(h5ad_in, h5ad_out, **kwargs)
    elif sc_type == "ATAC":
        return preprocess_scATAC(h5ad_in, h5ad_out, **kwargs)



