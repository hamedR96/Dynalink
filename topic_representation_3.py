import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from ctfidf import CTFIDFVectorizer


def doc2word(model,dic_ids,list_papers,wn):
    vecs=[]
    pure_words=[]
    for i in range(len(list_papers)):
        if list_papers[i] in dic_ids:
            index = dic_ids[list_papers[i]]
            vecs.append(model.model.docvecs[index])
    words=model.model.wv.most_similar(positive=vecs,topn=wn)
    for word in words:
        pure_words.append(word[0])
    return pure_words


def topic_dictionary(model,references_df):
    model_doc_df=pd.DataFrame({"internal_id":range(len(model.documents)),"abstract":model.documents})
    model_doc_df=model_doc_df.merge(references_df[["id","abstract"]], how='inner', on='abstract')
    dic_ids_df=model_doc_df[["id","internal_id"]]
    dic_ids= dic_ids_df.set_index('id')['internal_id'].to_dict()
    rev_dic_ids = {v: k for k, v in dic_ids.items()}
    return rev_dic_ids,dic_ids


def ctfidf_f(docs_per_class,docs_num):
    # Create bag of words
    count_vectorizer= CountVectorizer(ngram_range=(1, 2)).fit(docs_per_class.abstract)
    words= count_vectorizer.get_feature_names()
    # Create c-TF-IDF
    count= count_vectorizer.transform(docs_per_class.abstract)
    ctfidf= CTFIDFVectorizer().fit_transform(count, n_samples=docs_num).toarray()
    return words,ctfidf

def ctf_idf_topics(docs_per_class,words,ctfidf,num_terms):
    topics=[]
    for label in docs_per_class:
        topic=[]
        for index in ctfidf[label].argsort()[-num_terms:]:
            topic.append(words[index])
        topics.append(topic)
    return topics

