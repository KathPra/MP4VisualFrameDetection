# I Spy With My Little Eye: A Minimum Cost Multicut Investigation of Dataset Frames
Authors: Katharina Prasse, Isaac Bravo, Stefanie Walter, and Margret Keuper

Accepted at WACV 25 Applications Track.

You can find our paper on [arXiv](https://arxiv.org/abs/2412.01296).


## Overview

We propose to detect visual frames using a Minimum Cost Multicut Formulation.
To this end the following steps are necessary:

1. Embed the images using an embedding space of your choice (this also works for text!).
2. Compute pairwise cosine similarity scores between all inputs.
3. Map to a complete graph which nodes represent inputs and edges are weighted based on the inputs' cosine similarity.
4. Select hyperparameter *cal* from below table or compute your own.
5. Use MP solver to retrieve clusters.



## Set-Up

**Env:** The input for multicut solver needs to be in hdf. We share our environment in `FrameDet.yml`.

```
conda env create -f FrameDet.yml
```
<br>

**Image Embedding**: We mainly use the [transformers library](https://huggingface.co/docs/transformers/en/index) and install [dinov2](https://github.com/facebookresearch/dinov2) as explained on github.


**Multi-Cut**: We use the implementation by [B. Andres et al.](https://github.com/bjoern-andres/graph.git). We use [ccmake](https://cmake.org/cmake/help/latest/manual/ccmake.1.html) to compile the C++ code.


**Datasets**: We use [ImageNette, ImageWoof](https://github.com/fastai/imagenette?tab=readme-ov-file), and [ClimateTV](https://github.com/KathPra/Datasets_ClimateVisions). Please make sure to have one folder per dataset which can enter into the embedding script.

## Experiments
1. Embed input - fix paths within script
```
python emb/convnextv2.py
```
<br>

2. Compute cosine similarities
```
python graph_prep/cossim.py --dataset imagenette --model_config inception_resnet_v2 --embs path2embeddings/embs/ --setting ablation
```
<br>

3. Create graph
```
python python scripts/graph_mapping.py --dataset imagenette --model_config inception_resnet_v2 --embs path2embeddings/embs/ --split eval
```
<br>

4. Select hyperparameter *cal* from below table or compute your own. - fix paths within script
```
python graph_prep/ablate_bias.py
```
5. Use MP solver to retrieve clusters. Please provide input (-i) and output (-o) file paths and calibration term (-b).
```
cd ../graph
./solve-regular -i path2input_file/input_train.txt -o path2output_file/output_train.h5 -b 0.4
```

## Calibration Terms
In our work we ablate the calibration term cal on two datasets, ImageNette and ImageWoof. We share our ablated cal terms for use while encouraging authors to ablate their own cal terms when their use case differs from ours.

<!DOCTYPE html>
<html>
<head>
</head>
<body>
    <table>
        <tr>
            <th>Emb. model</th>
            <th>CLIP ViT-B-32</th>
            <th>DINOv2</th>
            <th>ConvNeXt V2</th>
            <th>ViT-B-32</th>
            <th>ResNet-50</th>
            <th>Inc.-ResNetv2</th>
            <th>VGG19-BN</th>
        </tr>
        <tr>
            <th>cal</th>
            <td>0.5</td>
            <td>0.6</td>
            <td>0.7</td>
            <td>0.7</td>
            <td>0.7</td>
            <td>0.5</td>
            <td>0.7</td>
        </tr>
    </table>
</body>
</html>

## Credits
We want to thank the authors of the Graph library for sharing such useful software. Moreover we extend our gratitude to all model architects for sharing invaluable embedding spaces.

```
@software{graph_mcmc,
  author = {Andres, Bjoern and Ibeling, Duligur and Kalofolias, Giannis and Keuper, Margret and Lange, Jan-Hendrik and Levinkov, Evgeny and Matten, Mark and Rempfler, Markus},
  title = {Graphs and Graph Algorithms in C++},
  url = {\url{http://www.andres.sc/graph.html}},
  date = {2024-07-01},
  year={2016},
  publisher={GitHub},
  howpublished = {\url{http://www.andres.sc/graph.html}},
}

@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={Transactions on Machine Learning Research Journal},
  year={2024}
}

@inproceedings{woo2023convnext,
  title={Convnext v2: Co-designing and scaling convnets with masked autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Citation
If you use our work, please cite us:
```
@InProceedings{Prasse_2025_WACV,
    author    = {Prasse, Katharina and Bravo, Isaac and Walter, Stefanie and Keuper, Margret},
    title     = {I Spy with My Little Eye A Minimum Cost Multicut Investigation of Dataset Frames},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2134-2143}
}
```
