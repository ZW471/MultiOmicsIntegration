# utils/data_utils.py
import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph
from datasets.preprocess import preprocess_scRNA, preprocess_ADT, preprocess_scATAC

dataset_config = {
    "BM-CITE": {
        "modalities": ["ADT", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad"
    },
    "LUNG-CITE": {
        "modalities": ["ADT", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad",
        "pca_components": {
            "ADT": 52
        }
    },
    "PBMC-DOGMA": {
        "modalities": ["ADT", "ATAC", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad",
        "modality_map": {"adt": "ADT", "atac": "ATAC"}  # Handle lowercase
    },
    "PBMC-Multiome": {
        "modalities": ["Peak", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad",
        "modality_map": {"Peak": "Peaks"}  # Handle plural forms
    },
    "PBMC-TEA": {
        "modalities": ["ADT", "ATAC", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad"
    },
    "Skin-SHARE": {
        "modalities": ["Peaks", "RNA"],
        "file_pattern": "{dataset}_{modality}.h5ad"
    }
}


def create_hetero_graph(data_dict, config, device):
    """Create PyG HeteroData object with KNN graphs"""
    processed = {m: {'x': torch.tensor(data_dict[m].X, dtype=torch.float)} 
                for m in config["modalities"]}
    data = HeteroData(processed)
    
    # Create cell node features
    data['cell'].x = torch.cat([data[m].x for m in config["modalities"]], dim=1)
    
    # Create KNN graphs
    data = data.to(device)
    for m in config["modalities"]:
        data['cell', m, 'cell'].edge_index = knn_graph(
            data[m].x, k=10, cosine=True, num_workers=16
        )
    return data.cpu()

def load_dataset(dataset_name, base_dir, device):
    config = dataset_config[dataset_name]
    data_dict = {}
    
    for modality in config["modalities"]:
        # Get filename pattern from config
        fname = config["file_pattern"].format(
            dataset=dataset_name,
            modality=config.get("modality_map", {}).get(modality, modality))
        
        input_path = os.path.join(base_dir, fname)
        
        # Preprocessing
        if modality == "RNA":
            data_dict[modality] = preprocess_scRNA(input_path)
        elif modality == "ADT":
            data_dict[modality] = preprocess_ADT(
                input_path,
                n_pcs=config.get("pca_components", {}).get(modality, None))
        elif modality == "ATAC":
            data_dict[modality] = preprocess_scATAC(input_path)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    return create_hetero_graph(data_dict, config, device), data_dict