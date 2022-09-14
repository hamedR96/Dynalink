import networkx as nx
from node2vec import Node2Vec

G=nx.read_graphml("citation_graph.gz")

node2vec = Node2Vec(G, dimensions=100, walk_length=100, num_walks=1000, workers=8)

n2v_model = node2vec.fit(window=10, min_count=1, batch_words=4)

n2v_model.wv.save_word2vec_format("EMBEDDING_FILENAME")
n2v_model.save("EMBEDDING_MODEL_FILENAME")