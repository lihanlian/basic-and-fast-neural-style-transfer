import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from fast_nst_transformer_net import TransformerNet
from fast_nst_vgg_net import VggNet
import utils
import os

utils.set_seed(42) # set the seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration

class CartoonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]  # List all .png or .jpg files
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((720, 720)),
    transforms.CenterCrop(720),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

normmalize_img = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_img(path, shape):
    img = Image.open(path).convert('RGB')

    in_transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor(),  # this normalize the PIL image to range [0, 1]
                                    transforms.Lambda(lambda x: x.mul(255)) # sacle image back to the range [0, 255])
                                    ])
    '''
    in_transform makes img has size of specified shape (in this case: (3, 720, 720))
    Pytorch convention: [batch size, channels, height, width]
    add batch size at the beginning to make img has size (1, 3, 720, 720)
    '''

    img = in_transform(img)
    img = img[:3,:,:].unsqueeze(0)
    return img.to(device)

# Initialize the dataset and dataloader
root_dir = './icartoonface'  # Change this to your dataset path
cartoon_dataset = CartoonDataset(root_dir=root_dir, transform=transform)
batch_size = 2
train_loader = DataLoader(cartoon_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize network and loss   
transformer_net = TransformerNet().to(device)
vgg_net = VggNet().to(device)

# load style image and store its feature gram matrices.
style_img_name = 'monet'
style_img = load_img(f'./figs/input_style/{style_img_name}.jpg',shape=(720,720)).to(device)  # Define load_image to preprocess the style image
style_img.div_(255.0)
style_img = normmalize_img(style_img)
style_img_features = vgg_net.forward(style_img)
# Return a dictionary with layer as the key, and its gram_matrix as values (using dictionary comprehension)
style_img_gram = {
    layer: utils.get_gram_matrix_fast_nst(style_img_features[layer]).repeat(batch_size, 1, 1)
    for layer in style_img_features
}

# Define optimizer
optimizer = optim.Adam(transformer_net.parameters(), lr=0.001)
mse_loss = torch.nn.MSELoss()

# Training loop setup
num_epochs = 1
total_batches = len(train_loader)
percent_interval = total_batches // 10
content_weight = 1; style_weight = 50000
print(f'Using device: {device}, style image: {style_img_name}, content weight(a): {content_weight}, style weight(b): {style_weight}')

for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for i, content_img in enumerate(train_loader):

            optimizer.zero_grad()
            # content_img: [0, 255], inference_img: [0, 255]
            content_img = content_img.to(device)
            inference_img = transformer_net(content_img) 

            # convert both content_img and inference_img to [0, 1] and then normalize
            content_img = content_img.div_(255.0)
            content_img = normmalize_img(content_img)
            inference_img = inference_img.div_(255.0)
            inference_img = normmalize_img(inference_img)
            
            # extract features using vgg_net
            content_img_features = vgg_net.forward(content_img) # extract feature informations of content images
            inference_img_features = vgg_net.forward(inference_img) # extract feature informations of inference images
            
            # calculate content loss (only use the output of layer 8)
            # ref: https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/neural_style.py
            content_loss = content_weight * mse_loss(inference_img_features['0'], content_img_features['0'])

            # calculate style loss
            style_loss = 0
            # breakpoint()
            # iterate output from each ReLU layer
            for layer_idx in style_img_features:
                # get gram matices of layer_idx from inference images 
                # _, c, h, w = style_img_features[layer_idx].size()
                inference_img_gram_layer_idx = utils.get_gram_matrix_fast_nst(inference_img_features[layer_idx])
                # get gram matices of layer_idx from style images 
                style_img_gram_layer_idx = style_img_gram[layer_idx]
                # get style loss of current layer_idx (by normalizing gram matrix difference)
                style_loss += mse_loss(inference_img_gram_layer_idx, style_img_gram_layer_idx)

            style_loss *= style_weight

            # calculate total loss
            loss = content_loss + style_loss
            loss.backward(retain_graph=True)

            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (i + 1))
            pbar.update(1)

            if (i+1) % percent_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

model_name = f'{style_img_name}_a{content_weight}b{style_weight}_e{num_epochs}'
torch.save(transformer_net.state_dict(), f'./models/{model_name}.pth')