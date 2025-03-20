# call this script with the following command:
# python cossim_unsup.py --dataset=imagenette --model_config=dinov2_vitb14_lc" --embs="feat_ext/"

############################################################################################################################

# load packages
import numpy as np
import torch
import tqdm
import argparse
import glob

torch.cuda.empty_cache()

def compute_cossim(data_points_dict):
    """
    Compute the pairwise cosine similarity between all data points stored in a dictionary.

    Parameters:
    data_points_dict (dict): A dictionary where keys are data IDs and values are data points.
    key_mapping (dict): A dictionary that contains the mapping of image ids to node ids.

    Returns:
    dict: A nested dictionary where the keys are data IDs and the values are dictionaries
          containing the distances to other data points.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Extract keys and values from the dictionary
    data_ids = list(data_points_dict.keys())
    data_points = torch.stack([data_points_dict[key] for key in data_ids]).squeeze(1).to(device)  # Move tensors to GPU

    # Normalize the data points
    norms = torch.norm(data_points, dim=1, keepdim=True)
    data_points = data_points / norms

    # Number of data points
    num_points = len(data_points)

    # Initialize a dictionary to store distances
    distances = {}
    distr = []

    # Compute pairwise cosine similarities
    for i in tqdm.tqdm(range(num_points)):
        distances[data_ids[i]] = {}
        for j in range(i+1, num_points):
            cos_sim = torch.dot(data_points[i],data_points[j]).item()
            distances[data_ids[i]][data_ids[j]] = cos_sim
            distr.append(cos_sim)

    return distances, distr, data_ids


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute cosine similarity between data points.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration name')
    parser.add_argument('--embs', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--setting', type=str, required=True, help='ablation (Train/Test) or evaluation (Test).')
    args = parser.parse_args()

    # Load embedding dictionary
    embs = torch.load(f"{args.embs}/{args.dataset}_{args.model_config}.torch")

    # if ablation setting, load 2 train/test split
    if args.setting == "ablation":
        # train test split
        train = glob.glob(f"../../Datasets/{args.dataset}/train/*/*.JPEG")
        train = [x.split("/")[-1] for x in train]
        val = glob.glob(f"../../Datasets/{args.dataset}/val/*/*.JPEG")
        val = [x.split("/")[-1] for x in val]

        embs_train = {k: v for k, v in embs.items() if k in train}
        embs_val = {k: v for k, v in embs.items() if k in val}

        # Compute distances
        distances_train, distr_val, data_ids_val = compute_cossim(embs_train)
        distances_val, distr_val, data_ids_val = compute_cossim(embs_val)
        print(len(distances_train))
        print(data_ids_val[0])

        # Save distances
        torch.save(distances_train, f"results/emb_dist/cossim_{args.dataset}_train_{args.model_config}.torch")
        torch.save(distances_train, f"results/emb_dist/cossim_{args.dataset}_val_{args.model_config}.torch")

    elif args.setting == "evaluation":
        # Compute distances
        distances, distr, data_ids = compute_cossim(embs)

        # Save distances
        torch.save(distances, f"results/emb_dist/cossim_{args.dataset}_{args.model_config}.torch")

        # Print statistics
        print(np.min(distr), np.max(distr))
        histo, bin_edges = np.histogram(distr)
        for i in range(len(histo)):
            print("bin ", bin_edges[i], "-", bin_edges[i + 1], "contains", histo[i], "samples")

    else:
        raise ValueError("Invalid setting. Choose either 'ablation' or 'evaluation'.")