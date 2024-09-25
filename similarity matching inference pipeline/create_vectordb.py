import faiss
import os
import argparse

import numpy as np
from tqdm import tqdm

from model.utils import load_model, get_embedding

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_arguments(): 

    parser = argparse.ArgumentParser(
        description="Create vector database."
    )

    parser.add_argument(
        "-r", "--refs", type=str, nargs="?", help="Path to reference images folder."
    )

    parser.add_argument(
        "-p", "--path_db", type=str, nargs="?", help="Path to the location that will store the vector database, default is ./vectordb.", default="./vectordb"
    )

    parser.add_argument(
        "-e", "--embedder", type=str, nargs="?", help="Path to the embedding model."
    )

    parser.add_argument(
        "-s", "--size", type=int, nargs="?", help="Size of embedding vector, default is 128.", default=128 
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Constants
    d = args.size 
    refs = args.refs 
    save_root = args.path_db

    os.makedirs(save_root, exist_ok=True)


    # Initializing embedding model 
    embedder = load_model(checkpoint_path=args.embedder, embeding_size=d)

    # Create index file 
    sets = os.listdir(refs)

    images = []
    for set_ in sets: 
        imgs = os.listdir(f"{refs}/{set_}")
        for img in imgs: 
            images.append(img)


    with open(f"{save_root}/index.txt", "w", encoding="utf-8") as f:
        for img in images: 
            string = f"{img}\n"
            f.writelines(string)


    # Create vector database
    embeddings = []

    for img in tqdm(images, desc="Creating database"):
        iset = img.split("_")[0]
        embedding = get_embedding(f"{refs}/{iset}/{img}", model=embedder)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    order = np.arange(embeddings.shape[0])


    index = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, order)


    faiss.write_index(index, f"{save_root}/vectors.bin")


