import torch
from PIL import Image
import random
import numpy as np

# set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_image(filename, data):
    img = data.clone().squeeze(0).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def get_gram_matrix_basic_nst(tensor):
    # b: batch size, c: number of channels, h: height, w: width
    (b, c, h, w) = tensor.size()
    features = tensor.view(b, c, h*w)
    gram = torch.bmm(features, features.transpose(1,2)) # transpose second and third dimension (h and w) 
    return gram

def get_gram_matrix_fast_nst(tensor):
    # b: batch size, c: number of channels, h: height, w: width
    (b, c, h, w) = tensor.size()
    features = tensor.view(b, c, h*w)
    gram = torch.bmm(features, features.transpose(1,2)) # transpose second and third dimension (h and w) 
    return gram / (c*h*w)
