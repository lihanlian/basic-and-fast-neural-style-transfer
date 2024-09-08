import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from PIL import Image
from fast_nst_transformer_net import TransformerNet
from fast_nst_vgg_net import VggNet
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, shape):
    img = Image.open(path).convert('RGB')


    reize_and_normalize = transforms.Compose([transforms.Resize(shape),
                                    # this normalize the PIL image to range [0, 1]
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.mul(255))
                                    # match imageNet distribution
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    # make img has size (3, 720, 720)
    img = reize_and_normalize(img)
    
    # Pytorch convention: [batch size, channels, height, width]
    # add batch size at the beginning to make img has size (1, 3, 720, 720)
    img = img[:3,:,:].unsqueeze(0)
    return img.to(device)



def apply_style(image_path, output_name, model_name):
    # Load pretrained TransformerNet model
    transform_net = TransformerNet().to(device)
    model_path = f'./models/{model_name}.pth'
    transform_net.load_state_dict(torch.load(model_path, weights_only=True))
    transform_net.eval()

    # Load content image
    content_img = load_image(image_path, (720,720))
    
    with torch.no_grad():
        output = transform_net(content_img).cpu()
    saved_path = f'./figs/output_fast_nst/{output_name}'
    utils.save_image(saved_path, output)
    # save_img(output, saved_path)  # Define save_image to save the stylized image

content_weight = 1; style_weight = 50000
num_epoch = 1
style_img_name = 'monet'
content_img_name = 'minion1'
model_name = f'{style_img_name}_a{content_weight}b{style_weight}_e{num_epoch}'
apply_style(image_path=f'./figs/input_content/{content_img_name}.jpg', 
            output_name=f'{content_img_name}_{style_img_name}_a{content_weight}b{style_weight}_e{num_epoch}.jpg',
            model_name=model_name)