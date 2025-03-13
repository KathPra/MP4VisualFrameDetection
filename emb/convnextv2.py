## Dataset paths - modify to match your setup
img_dir = "../../../Datasets/imagenette2/"

## Dataset name for saving - modify to match your setup
dataset = "imagenette"

## Model name (for loading and saving the model) 
model_name = "facebook/convnextv2-tiny-1k-224"

## Output file path - modify to match your setup
output_file = f"features/{dataset}_{model_name}.torch".replace("ok/c","ok_c")

############################################################################################################

## load packages
import torch
import glob 
from torchvision import transforms
from PIL import Image, ImageFile
import tqdm
import os
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch.nn as nn

# allow to load images that exceed max size
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the model
processor = AutoImageProcessor.from_pretrained(model_name)
model = ConvNextV2ForImageClassification.from_pretrained(model_name)
model.classifier = nn.Identity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
#print(model)

# Define a transform to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to compute and save embeddings
def compute_embeddings(img_dir, model,output_file):
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

        image = processor(images=image, return_tensors="pt")#.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model(**image.to(device)).logits
        #last_hidden_states = outputs.last_hidden_state

        embeddings[image_name] = embedding
    
    torch.save(embeddings, output_file)


def find_all_image_paths(root_dir):
    '''
    Input: the root directory of the dataset
    Output: a dictionary where keys are image names and values are image paths
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