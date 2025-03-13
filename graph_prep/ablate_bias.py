## conda env: vis_h5

# import packages
import h5py
import numpy as np
import glob
import torch

from data_prep import extract_data, create_img_clust_mapping, unify_naming, final_mapping_sup
from utils.metrics import cond_entropy, entropy
from utils.vis import plot_cond_entropy, plot_cond_entropy_train

dataset = "imagewoof"                                                                               ## adjust to dataset for which classes are available

# load mapping from img names to node ids (dict) key = img_file name, value = node id
key_mapping = torch.load(f"key_mapping_{dataset}_train.torch")                                            ## constructed using graph_mapping.py
train_imgs = list(key_mapping.keys())

# load data (repeat for all embedding models)
model_config, save_name = "ViT-B-32_openai", "CLIP ViT-B/32 (IN)"                                   ## specify embedding model
print(save_name)

# load clusterings created will all bias values (train)                                             ## adjust paths
b1 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b1.h5", "r")
b2 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b2.h5", "r")
b3 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b3.h5", "r")
b4 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b4.h5", "r")
b5 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b5.h5", "r")
b6 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b6.h5", "r")
b7 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b7.h5", "r")
b8 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b8.h5", "r")
b9 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_train_b9.h5", "r")

"""
## run once to map classes to numbers (based on dataset classes)
# map classes to numbers
labels_org = glob.glob(f"../../Datasets/{dataset}/train/*/*.JPEG")                                     ## adjust to dataset
labels_org = [l[l.find("/n")+1:] for l in labels_org]
classes = set([l[:l.find("/")] for l in labels_org])
class_num = dict(zip(classes, [i for i in range(len(classes))]))
torch.save(class_num, f"results/ablation/class_num_{dataset}_{model_config}_train.torch")   

# create class mapping from images(key) to classes (value)
class_dict = {c:[] for c in classes}
for img in labels_org:
    class_dict[img[:img.find("/")]].append(img[img.find("/")+1:])
class_mapping = {}
for c in class_dict:
    for img in class_dict[c]:
        class_mapping[img] = c
torch.save(class_mapping, f"results/class_mapping_{dataset}_{model_config}_train.torch")
"""
class_num = torch.load(f"results/ablation/class_num_{dataset}_{model_config}_train.torch")                ## after creation, use files
class_mapping = torch.load(f"results/class_mapping_{dataset}_{model_config}_train.torch")
class_org = [class_num[k] for k in class_mapping.values()]

# assess als calibration terms
print("b=0.1")
labels1, counter_labels1, energy1, rt1 = extract_data(b1)
clust_overview1, img_labels1,_ = create_img_clust_mapping(counter_labels1, labels1, key_mapping)
final_label_mapping_b1 = unify_naming(class_mapping, clust_overview1, class_num)
final_mapping1 = final_mapping_sup(clust_overview1, final_label_mapping_b1)

print("b=0.2")
labels2, counter_labels2, energy2, rt2 = extract_data(b2)
clust_overview2, img_labels2,_ = create_img_clust_mapping(counter_labels2, labels2, key_mapping)
final_label_mapping_b2 = unify_naming(class_mapping, clust_overview2, class_num)
final_mapping2 = final_mapping_sup(clust_overview2, final_label_mapping_b2)

print("b=0.3")
labels3, counter_labels3, energy3, rt3 = extract_data(b3)
clust_overview3, img_labels3,_ = create_img_clust_mapping(counter_labels3, labels3, key_mapping)
final_label_mapping_b3 = unify_naming(class_mapping, clust_overview3, class_num)
final_mapping3 = final_mapping_sup(clust_overview3, final_label_mapping_b3)

print("b=0.4")
labels4, counter_labels4, energy4, rt4 = extract_data(b4)
clust_overview4, img_labels4,_ = create_img_clust_mapping(counter_labels4, labels4, key_mapping)
final_label_mapping_b4 = unify_naming(class_mapping, clust_overview4, class_num)
final_mapping4 = final_mapping_sup(clust_overview4, final_label_mapping_b4)

print("b=0.5")
labels5, counter_labels5, energy5, rt5 = extract_data(b5)
clust_overview5, img_labels5,_ = create_img_clust_mapping(counter_labels5, labels5, key_mapping)
final_label_mapping_b5 = unify_naming(class_mapping, clust_overview5, class_num)
final_mapping5 = final_mapping_sup(clust_overview5, final_label_mapping_b5)

print("b=0.6")
labels6, counter_labels6, energy6, rt6 = extract_data(b6)
clust_overview6, img_labels6,_ = create_img_clust_mapping(counter_labels6, labels6, key_mapping)
final_label_mapping_b6 = unify_naming(class_mapping, clust_overview6, class_num)
final_mapping6 = final_mapping_sup(clust_overview6, final_label_mapping_b6)

print("b=0.7")
labels7, counter_labels7, energy7, rt7 = extract_data(b7)
clust_overview7, img_labels7,_ = create_img_clust_mapping(counter_labels7, labels7, key_mapping)
final_label_mapping_b7 = unify_naming(class_mapping, clust_overview7, class_num)
final_mapping7 = final_mapping_sup(clust_overview7, final_label_mapping_b7) 

print("b=0.8")
labels8, counter_labels8, energy8, rt8 = extract_data(b8)
clust_overview8, img_labels8,_ = create_img_clust_mapping(counter_labels8, labels8, key_mapping)
final_label_mapping_b8 = unify_naming(class_mapping, clust_overview8, class_num)
final_mapping8 = final_mapping_sup(clust_overview8, final_label_mapping_b8)

print("b=0.9")
labels9, counter_labels9, energy9, rt9 = extract_data(b9)
clust_overview9, img_labels9,_ = create_img_clust_mapping(counter_labels9, labels9, key_mapping)
final_label_mapping_b9 = unify_naming(class_mapping, clust_overview9, class_num)
final_mapping9 = final_mapping_sup(clust_overview9, final_label_mapping_b9)



print("entropy")

e_b1 = print("b=0.1", entropy(final_mapping1))
e_b2 = print("b=0.2", entropy(final_mapping2))
e_b3 = print("b=0.3", entropy(final_mapping3))
e_b4 = print("b=0.4", entropy(final_mapping4))
e_b5 = print("b=0.5", entropy(final_mapping5))
e_b6 = print("b=0.6", entropy(final_mapping6))
e_b7 = print("b=0.7", entropy(final_mapping7))
e_b8 = print("b=0.8", entropy(final_mapping8))
e_b9 = print("b=0.9", entropy(final_mapping9))
e_org = print("annotation", entropy(class_org))

print("VI")
print(len(final_mapping1), len(class_org))
H_b1_org, H_org_b1, VI_1 = cond_entropy(final_mapping1, class_org)
H_b2_org, H_org_b2, VI_2 = cond_entropy(final_mapping2, class_org)
H_b3_org, H_org_b3, VI_3 = cond_entropy(final_mapping3, class_org)
H_b4_org, H_org_b4, VI_4 = cond_entropy(final_mapping4, class_org)
H_b5_org, H_org_b5, VI_5 = cond_entropy(final_mapping5, class_org)
H_b6_org, H_org_b6, VI_6 = cond_entropy(final_mapping6, class_org)
H_b7_org, H_org_b7, VI_7 = cond_entropy(final_mapping7, class_org)
H_b8_org, H_org_b8, VI_8 = cond_entropy(final_mapping8, class_org)
H_b9_org, H_org_b9, VI_9 = cond_entropy(final_mapping9, class_org)

print("VI")
print("B=0.1", VI_1)
print("B=0.2", VI_2)
print("B=0.3", VI_3)
print("B=0.4", VI_4)
print("B=0.5", VI_5)
print("B=0.6", VI_6)
print("B=0.7", VI_7)
print("B=0.8", VI_8)
print("B=0.9", VI_9)

bias = np.array([[H_b1_org, H_b2_org, H_b3_org, H_b4_org, H_b5_org, H_b6_org, H_b7_org, H_b8_org, H_b9_org],[H_org_b1, H_org_b2, H_org_b3, H_org_b4, H_org_b5, H_org_b6, H_org_b7, H_org_b8, H_org_b9]])
labels = np.array(["c=0.1", "c=0.2", "c=0.3", "c=0.4", "c=0.5", "c=0.6", "c=0.7", "c=0.8", "c=0.9"])

plot_cond_entropy_train(bias, dataset, model_config, save_name, labels)

################################ VERIFY using validation set ############################################
# load clusterings created will all bias values (test)                                             ## adjust paths
b1 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b1.h5", "r")
b2 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b2.h5", "r")
b3 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b3.h5", "r")
b4 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b4.h5", "r")
b5 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b5.h5", "r")
b6 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b6.h5", "r")
b7 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b7.h5", "r")
b8 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b8.h5", "r")
b9 = h5py.File(f"results/multicut/{dataset}_{model_config}/output_test_b9.h5", "r")
