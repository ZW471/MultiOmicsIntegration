import subprocess

file_names = ['LUNG-CITE', 'BM-CITE', 'PBMC-DOGMA', 'PBMC-Multiome', 'PBMC-TEA', 'Skin-SHARE']

for file_name in file_names:
    subprocess.run(["Rscript", "mojitoo_data_to_anndata.R", file_name], check=True)
