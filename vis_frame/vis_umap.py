## conda env: umap

import umap
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# load embeddings and normalize embeddings
model_name = "example" #### MODIFY MODEL NAME
embs = torch.load("feat_ext/tvjan_dinov2_vitb14_lc.torch", weights_only = False) ### MODIFY PATH
embs = torch.stack([embs[key] for key in embs.keys()]).squeeze(1).cpu()
embs_norm = embs / embs.norm(dim=-1, keepdim=True)

# reduce dimensionality
reducer = umap.UMAP()
embs_reduced = reducer.fit_transform(embs_norm)

# plot umap projection of embedding space
plt.scatter(embs_reduced[:, 0], embs_reduced[:, 1], c="blue", label = model_name, linewidths=0)
plt.title(f'UMAP projection of {model_name} embeddings')
plt.tight_layout()
plt.ylim(-10, 20)
plt.xlim(-15, 20)
plt.savefig(f"results/vis/umap_{model_name}.pdf")
plt.close()

"""
embs1 = torch.load("feat_ext/tvjan_dinov2_vitb14_lc.torch", weights_only = False) ### MODIFY PATH
embs1 = torch.stack([embs1[key] for key in embs1.keys()]).squeeze(1).cpu()
embs_norm1 = embs1 / embs1.norm(dim=-1, keepdim=True)
reducer = umap.UMAP()
embs_reduced1 = reducer.fit_transform(embs_norm1)

embs2 = torch.load("feat_ext/tvjan_dinov2_vitb14_lc.torch", weights_only = False) ### MODIFY PATH
embs2 = torch.stack([embs2[key] for key in embs2.keys()]).squeeze(1).cpu()
embs_norm2 = embs2 / embs2.norm(dim=-1, keepdim=True)
reducer = umap.UMAP()
embs_reduced2 = reducer.fit_transform(embs_norm2)

# compare stats between two embs
embs_reduced_comp = torch.nn.functional.cosine_similarity(torch.tensor(embs_reduced1), torch.tensor(embs_reduced2), dim=1)
print(torch.mean(embs_reduced_comp))
print(torch.std(embs_reduced_comp))
print(torch.min(embs_reduced_comp))
print(torch.min(embs_reduced_comp))
# how many have cosine similarity smaller 0
print(torch.sum(embs_reduced_comp < 0))
"""
"""
# plot umap projection of 2 embedding spaces to show overlap
comp_name = "vision foundation models"
plt.scatter(embs_reduced1[:, 0], embs_reduced1[:, 1], c="blue", label = model_name1, linewidths=0)
plt.scatter(embs_reduced2[:, 0], embs_reduced2[:, 1], c="blue", label = model_name2, linewidths=0)
plt.title(f'UMAP projection of {comp_name}')
plt.tight_layout()
plt.ylim(-10, 20)
plt.xlim(-15, 20)
plt.legend()
plt.savefig(f"results/vis/umap_{comp_name}.pdf")
plt.close()
"""
