import subprocess

file_names = ['PBMC-DOGMA']

for file_name in file_names:
    subprocess.run(["Rscript", "mojitoo_data_to_anndata.R", file_name], check=True)
