import torch
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

from model.similarity_module import LightContrastiveEmbedding

to_tensor = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path, embeding_size, model_type): 
    if model_type == "light":
        model = LightContrastiveEmbedding(embedding_size=embeding_size)
    else: 
        model = ContrastiveEmbedding(embedding_size=embeding_size)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    
    df_state_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        name = k[6:] 
        df_state_dict[name]=v

    model.load_state_dict(df_state_dict)
    return model    

def get_embedding(img, model): 
    model.eval()
    input_img = Image.open(img)
    input_img = to_tensor(input_img)

    with torch.no_grad(): 
        embedding = F.normalize(model(input_img.unsqueeze(0)), p=2, dim=1)

    return embedding.squeeze(0).numpy()

def most_common(lst):
    return max(set(lst), key=lst.count)

