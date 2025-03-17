import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def compute_ari_nmi(adata,
                    cluster_key="louvain",
                    truth_key="celltype"):
    """
    Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
    between predicted clusters and ground-truth labels.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object after clustering.
    cluster_key : str
        adata.obs key with the clustering labels (e.g., 'louvain').
    truth_key : str
        adata.obs key with ground-truth labels (e.g., 'celltype').
    
    Returns
    -------
    (float, float)
        ARI and NMI values.
    """
    pred_labels = adata.obs[cluster_key].tolist()
    true_labels = adata.obs[truth_key].tolist()
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi


def compute_silhouette(adata,
                       embedding_key="mojitoo",
                       cluster_key="louvain"):
    """
    Compute the silhouette score for clusters in the integrated embedding.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with an integrated embedding in adata.obsm[embedding_key].
    embedding_key : str
        adata.obsm key for the integrated embedding (e.g., 'mojitoo').
    cluster_key : str
        adata.obs key with the clustering labels (e.g., 'louvain').
    
    Returns
    -------
    float
        The silhouette score (higher is better).
    """
    X = adata.obsm[embedding_key]
    labels = adata.obs[cluster_key].values
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(X, labels, metric='euclidean')


def compute_structure_preservation(
    adata,
    integrated_key="mojitoo",
    original_keys=("pca", "apca")
):
    """
    Compute structure-preservation scores by measuring the Pearson correlation
    between pairwise distances in the integrated space and each original modality's space.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing both the integrated embedding and original modality embeddings.
    integrated_key : str
        adata.obsm key for the integrated embedding (e.g., 'mojitoo').
    original_keys : tuple or list of str
        A list of adata.obsm keys for the original modalities (e.g., ['pca','apca']).
    
    Returns
    -------
    dict
        A dictionary with {modality_key: (correlation, pvalue)} entries.
    """
    Z = adata.obsm[integrated_key]  # integrated representation
    dist_integ = pdist(Z, metric='euclidean')

    results = {}
    for key in original_keys:
        if key not in adata.obsm:
            continue
        X_mod = adata.obsm[key]
        dist_mod = pdist(X_mod, metric='euclidean')
        r, p = pearsonr(dist_integ, dist_mod)
        results[key] = (r, p)

    return results
