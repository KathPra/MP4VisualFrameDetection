## Dataset paths - modify to match your setup
img_dir = "../../../Datasets/imagenette2/"

## Dataset name for saving - modify to match your setup
dataset = "imagenette"

## Model name (for loading and saving the model) 
model_name = "dinov2_vitb14_lc"

## Output file path - modify to match your setup
output_file = f"../feat_ext/embs/{dataset}_{model_name}.torch".replace("ok/c","ok_c")

############################################################################################################

# load packages
import torch
from PIL import Image, ImageFile
import tqdm
import os
import torch.nn as nn
from dinov2.data.transforms import make_classification_eval_transform

# allow to load images that exceed max size
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load DINOv2
model = torch.hub.load('facebookresearch/dinov2', model_name)
# replace classification head by indentity to retrieve embeddings
model.linear_head = torch.nn.Identity()
transform = make_classification_eval_transform()

# prepare embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to compute and save embeddings
def compute_embeddings(img_dir, model, transform, output_file):
    '''
    Input: Dataset path, model, output file path
    Output: Saves embeddings to output file
    '''
    embeddings = {}
    for image_name in tqdm.tqdm(img_dir.keys()):
        image_path = img_dir[image_name]
        image = Image.open(image_path)

        # Handle images with transparency
        if image.mode in ('RGBA', "P") and len(image.getbands())==4: 
            # convnext cannot handle RGBA inputs
            image_org = image.copy()
            image = Image.new("RGB", image_org.size, (255, 255, 255))
            image.paste(image_org, mask=image_org.split()[3])
            #image = image.convert("RGBA")
        else: image = image.convert("RGB")

        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model(image.to(device))#.logits
        #last_hidden_states = outputs.last_hidden_state

        embeddings[image_name] = embedding
    
    torch.save(embeddings, output_file)


def find_all_image_paths(root_dir):
    '''
    Input: the root directory of the dataset
    Output: A dictionary where keys are image names and values are image paths
    '''
    image_paths = {}
    for dirpath, _, filenames in os.walk(root_dir): 
        img_filenames = list(set(filenames))
        for filename in img_filenames:
            image_paths[filename] = os.path.join(dirpath, filename)
    print(len(image_paths))
    return image_paths



## Execute function calls
img_path_dir = find_all_image_paths(img_dir)
# check that all images have been found
print(len(img_path_dir))

compute_embeddings(img_path_dir, model, output_file)