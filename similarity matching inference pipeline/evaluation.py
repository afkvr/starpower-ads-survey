import os 
import argparse

import faiss
from tqdm import tqdm
import torch

from torchmetrics import F1Score, Accuracy

from inference import predicition
from model.utils import load_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_arguments(): 

    parser = argparse.ArgumentParser(
        description="Evaluating embedding model's few-shot learning ability."
    )

    parser.add_argument(
        "-t", "--test", type=str, nargs="?", help="Path to test images folder."
    )

    parser.add_argument(
        "-d", "--database", type=str, nargs="?", help="Path to the vector database, default is ./vectordb.", default="./vectordb"
    )

    parser.add_argument(
        "-e", "--embedder", type=str, nargs="?", help="Path to the embedding model."
    )

    parser.add_argument(
        "-s", "--size", type=int, nargs="?", help="Size of the embedding vector, default is 128.", default=128 
    )

    parser.add_argument(
        "-k", "--k_neighbors", type=int, nargs="?", help="Number of nearst neighbors to perform majority vote, default is 3.", default=3
    )

    return parser.parse_args()


if __name__=="__main__":

    args = parse_arguments()

    # Constants
    d = args.size 
    test_root = args.test 
    db_root = args.database
    k = args.k_neighbors


    # Initializing embedding model 
    embedder = load_model(checkpoint_path=args.embedder, embeding_size=d)

    # Create index file 
    with open (f"{db_root}/index.txt", "r", encoding="utf-8") as f: 
        index_mapping = f.read().splitlines()

    vectordb = faiss.read_index(f"{db_root}/vectors.bin")

    # Load in test images
    sets = os.listdir(test_root)

    images = [] 

    for set_ in sets: 
        imgs = os.listdir(f"{test_root}/{set_}")
        for img in imgs: 
            images.append(img)


    # Evaluation
        #Defining evaluation metrics
    LABEL2ID = {key: value for value, key in enumerate(sets)}
    num_classes = len(sets)
    avg="micro"
    task="multiclass"
    f1  = F1Score(task=task, average=avg, num_classes=num_classes)
    acc = Accuracy(task=task, average=avg, num_classes=num_classes)

    labels = [] 
    preds = []
    for img in tqdm(images, desc="ETA"):

        iset = img.split("_")[0]
        abs_path = f"{test_root}/{iset}/{img}"

        pred = predicition(image=abs_path, embedder=embedder, index_mapping=index_mapping, vectordb=vectordb, k=k)

        labels.append(LABEL2ID[iset])
        preds.append(LABEL2ID[pred])

    labels = torch.tensor(labels)
    preds = torch.tensor(preds)

    print(f"Top {k} accuracy:{acc(preds, labels).item(): .2f}")
    print(f"Top {k} f1-score:{f1(preds, labels).item(): .2f}")
