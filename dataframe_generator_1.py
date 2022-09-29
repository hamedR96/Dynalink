import pandas as pd

def topic_pandas_generator(model,references_df):
    model_doc_df=pd.DataFrame({"internal_topic_id":range(len(model.documents)),"abstract":model.documents})
    model_doc_df=model_doc_df.merge(references_df[["id","abstract"]], how='inner', on='abstract')
    dic_ids_df=model_doc_df[["id","internal_topic_id"]]
    return dic_ids_df

def d2v_pandas_generator(model,references_df):
    df=topic_pandas_generator(model,references_df)
    df["d2v"]=df.apply(lambda row: model.model.docvecs[row["internal_topic_id"]], axis=1 )
    return df

def n2v_df(row):
    try:
        return n2v_model.wv.get_vector(row)
    except:
        return pd.NaT