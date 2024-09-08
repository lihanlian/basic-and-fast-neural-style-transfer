import torch
import torch.nn as nn
import torch.optim as optimization
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
import utils

utils.set_seed(0) # set random seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set the device for image tensor storage

def load_img(path, shape=None):
    img = Image.open(path).convert('RGB')


    reize_and_normalize = transforms.Compose([transforms.Resize(shape),
                                    # this normalize the PIL image to range [0, 1]
                                        transforms.ToTensor(),
                                    # match imageNet distribution
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # make img has size (3, 720, 720)
    img = reize_and_normalize(img)
    
    # Pytorch convention: [batch size, channels, height, width]
    # add batch size at the beginning to make img has size (1, 3, 720, 720)
    img = img[:3,:,:].unsqueeze(0)
    return img.to(device)

def save_img(output, iteration):
  # image needs to be denormalized first
  denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
  # removes the additional dimension
  output_img = output.clone().detach().cpu().squeeze()
  output_img = denormalization(output_img).clamp(0, 1)
  save_image(output_img, f'./figs/output_basic_nst/{style_img_name}/{content_img_name}_a{alpha}b{beta}_{iteration}.jpg')

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # set index of layers of interest for computing loss
        self.select_features = ['0', '5', '10', '19', '28']

        # load pre-trained VGG19 model and extracts only feature extraction layers (conv layers),
        # excluding the fully connected layers
        self.vgg = models.vgg19(pretrained = True).features
        print('VGG-19:')
        print(self.vgg)
        print(20*'***')

    def forward(self, x):
        # store output of feature maps from selected layers
        features = []
        for name, layer in self.vgg._modules.items():
            # apply current layer's operation
            x = layer(x)
            if name in self.select_features:
                features.append(x)
        return features
    
# load content and style images
content_img_name = 'minion1'
content_img = load_img(f'./figs/input_content/{content_img_name}.jpg', shape=(720,720))
style_img_name = 'picasso'
style_img = load_img(f'./figs/input_style/{style_img_name}.jpg', shape=(720,720))


output_img = content_img.clone().requires_grad_(True) # set content image as the starting point for optimization
vgg = VGG().to(device).eval() # set vgg to evalution mode since it is just used for feature extraction

lr = 0.001
optimizer = optimization.Adam([output_img], lr=lr)
mse_loss = torch.nn.MSELoss()
alpha = 1 # content weight
beta = 100000 # style weight
num_steps = 25000
print(f'Using device: {device}, content image: {content_img_name}, style image: {style_img_name}, content weight(a): {alpha}, style weight(b): {beta}')

for i in tqdm(range(num_steps + 1)):
    optimizer.zero_grad()

    output_img_features_all = vgg(output_img)
    content_img_features_all = vgg(content_img)
    style_img_features_all = vgg(style_img)

    # calculate content loss using features from the 3rd selected layer (conv4_2)
    content_loss = alpha * mse_loss(output_img_features_all[2], content_img_features_all[2])

    # calculate style loss
    style_loss = 0
    for output_img_feature, style_img_feature, layer in zip(output_img_features_all, style_img_features_all, vgg.select_features):
        _, c, h, w = output_img_feature.size()
        output_img_gram = utils.get_gram_matrix_basic_nst(output_img_feature)
        style_img_gram = utils.get_gram_matrix_basic_nst(style_img_feature)
        # normalize the gram matrices for style loss
        style_loss += mse_loss(output_img_gram, style_img_gram) / (c*h*w)
    style_loss *= beta     

    total_loss = content_loss + style_loss 
    total_loss.backward()
    optimizer.step()

    # log loss info and store intermediate images
    if i % 1000 == 0:
        print('Total loss: ', total_loss.item())
    if i % 5000 == 0 and i != 0:
        save_img(output_img, i)
    