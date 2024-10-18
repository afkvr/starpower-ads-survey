import faiss
import os
from model.utils import load_model, get_embedding, most_common

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def predicition(image, embedder, index_mapping, vectordb , k=3):

    embedder = embedder

    q = get_embedding(img=image, model=embedder).reshape(1, -1)

    D, I = vectordb.search(q, k)
    res = I.squeeze().tolist()
    if k == 1: 
        return index_mapping[res].split("_")[0], index_mapping[res]
    
    neighbors = [index_mapping[r] for r in res]
    pred = most_common([neighbor.split("_")[0] for neighbor in neighbors])

    return pred, neighbors

# Example: 
if __name__ =="__main__":

    image = "path_to_image"
    embedder = load_model(checkpoint_path="./checkpoints/....", embeding_size=128, model_type="light") #Model type can be light or normal

    # Load in index file and vector database
    with open (".../vectordb/index.txt", "r", encoding="utf-8") as f: 
        index_mapping = f.read().splitlines()

    vectordb = faiss.read_index(".../vectordb/vectors.bin")

    pred = predicition(image=image, embedder=embedder, index_mapping=index_mapping, vectordb=vectordb, k=3)
