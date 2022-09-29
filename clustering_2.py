import umap
import hdbscan
import pandas as pd

def clustering(df,umap_args,hdbscan_args):
    umap_model = umap.UMAP(**umap_args).fit(df)
    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)
    return cluster