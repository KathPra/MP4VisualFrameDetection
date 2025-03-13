## conda env: env_prep

# import packages
import torch
import numpy as np
import argparse
import tqdm


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute cosine similarity between data points.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration name')
    parser.add_argument('--dist_metric', type=str, required=True, help='Which distance metric to use')
    parser.add_argument('--embs', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--setting', type=str, required=True, help='Ablation (Train/Test) or Evaluation (Test).')
    args = parser.parse_args()

# load data
dist_metric = args.dist_metric
model_config = args.model_config
dataset = args.dataset
dist = torch.load(f"emb_dist/{dist_metric}_{dataset}_{model_config}.torch")

# create dict for mapping img_ids to node_ids
# Load embedding dictionary
embs = torch.load(f"{args.embs}/{args.dataset}_{args.model_config}.torch")
img_ids = list(embs.keys())
node_ids = [i for i in range(len(img_ids))]
key_mapping = dict(zip(img_ids, node_ids))
torch.save(key_mapping, f"key_mapping_{args.dataset}.torch")
embs = None
img_ids = None
node_ids = None
key_mapping = torch.load(f"key_mapping_{args.dataset}.torch")
print("key mapping created")

# create edges between nodes (images) that have a high similarity (w.r.t. threshold)
edge_array = []
weight_array = []

# l2 dist = 0.9 2.6 %, 0.95 (5.57 %), 1.0 (11.49 %), 1.05 (22.34 %), 1.1 (38.86 %)
#threshold = 1.0
for k,v in tqdm.tqdm(dist.items()):
    for t,z in v.items():
        edge_array.append(np.array([int(key_mapping[k]), int(key_mapping[t])], dtype=int))
        weight_array.append(z)

edge_array = np.asarray(edge_array, dtype=int)
weight_array = np.asarray(weight_array, dtype=float)
# scale array to the interval [0,1]
weight_array_norm = (weight_array-np.min(weight_array))/(np.max(weight_array)-np.min(weight_array))
print(edge_array.shape, weight_array.shape)
weight_array = None
print(weight_array_norm.shape)
final_array = np.concatenate((edge_array, weight_array_norm[:, None]), axis=1)
print(final_array.shape)

#print(edge_array, weight_array)

with open(f"multicut/{dist_metric}_{dataset}_{model_config}/input.txt", "w") as f:
    f.write(f"{len(key_mapping)} {len(edge_array)}\n")
    np.savetxt(f, final_array, encoding=None, fmt='%d %d %.6f')

## print stats
print(np.min(weight_array_norm), np.max(weight_array_norm))
histo, bin_edges = np.histogram(weight_array_norm)
for i in range(len(histo)):
    print("bin ", bin_edges[i], "-", bin_edges[i + 1], "contains", histo[i], "samples")
