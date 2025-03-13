import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

def plot_clust(all_clust_len, dataset):
    """
    Input:
    all_clust_len: list of lists, each list contains the cluster sizes for one model
    dataset: str, name of the dataset

    Returns:
    Figure saved showing boxplotu of cluster sizes for each model and class size (red line)
    """
    plt.close()
    labels = ["ConvNeXtv2", "CLIP ViT-B/32", "DINOv2", "ViT-B/32", "RN-50"] # list all embedding spaces included in the comparison

    # plot
    fig, ax = plt.subplots(constrained_layout=True)
    VP = ax.boxplot(all_clust_len,"rs", positions=[2, 4, 6,8,10], widths=1, patch_artist=True,  # Adjust the positions to the labels
                    showmeans=True, showfliers=True) 
    
    ax.set_xticklabels(labels, fontsize=16, rotation=45, ha="right")  # Rotate x-axis labels and set font size
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)  # Set y-tick labels font size

    # this adds a red line at the class size
    ax.axhline(y=1350, color="r")
    ax.set_ylim(-10, 4000)                                                                   # Adjust the y-axis limits according to the data

    plt.savefig(f"results/vis/model_comp_{dataset}_cluster_sizes.pdf")
    plt.close()

def plot_clust_unsup(all_clust_len, dataset):
    """
    Input:
    all_clust_len: list of lists, each list contains the cluster sizes for one model
    dataset: str, name of the dataset

    Returns:
    Figure saved showing boxplotu of cluster sizes for each model
    """
    plt.close()
    labels = ["ConvNeXtv2", "CLIP ViT-B/32", "DINOv2", "ViT-B/32", "RN-50"]

    # plot
    fig, ax = plt.subplots(figsize=(12, 6),constrained_layout=True)
    VP = ax.boxplot(all_clust_len,"rs", widths=1, patch_artist=True,
                    showmeans=True, showfliers=True, vert=False)
    

    ax.set_xticklabels(ax.get_xticks(), fontsize=16)#, rotation=45, ha="right")  # Rotate x-axis labels and set font size
    ax.set_yticklabels(labels, fontsize=16)  # Set y-tick labels font size

    # Format x-axis ticks to remove digits after the comma
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

    plt.savefig(f"vis/model_comp_{dataset}_cluster_sizes.pdf")
    plt.close()



def func(x, a, b):
    """
    Function to fit an expontential curve to the data
    """
    return a * np.exp(-b * x)

def plot_cond_entropy(all_cond_entropy, dataset, model, save_name, labels):
    """
    all_cond_entropy: list, contains the conditional entropy values for one model
    
    create a plot with points(x,y) which are the two conditional entropy values
    """
    # define plot
    plt.axis("scaled")
    plt.ylim(0,7)
    plt.xlim(0,4)
    plt.yticks(np.arange(0,8,1))
    plt.xticks(np.arange(0,5,1))

    #fit curve
    popt, pcov = curve_fit(func, all_cond_entropy[0], all_cond_entropy[1])
    print(popt, pcov)
    plot_points = np.arange(0, 5, 0.3)
    plt.plot(plot_points, func(plot_points, *popt),'--k', alpha=0.3)

    # Fit curve
    #ylog_data = np.log(all_cond_entropy[1])
    #curve_fit = np.polyfit(all_cond_entropy[0], ylog_data, 1)
    #print(curve_fit)
    #y_new = np.exp(curve_fit[0]) * np.exp(curve_fit[1]*all_cond_entropy[0])
    #plt.plot(all_cond_entropy[0], y_new, "--k", alpha=0.3)

    # plot points in all_cond_entropy (x,y)
    plt.plot(all_cond_entropy[0],all_cond_entropy[1], 'o',  zorder=10, clip_on=False)
    

    """
    # Check if any two points points are very near to each other
    min_dist = 0.5
    special_points = []
    normal_points = []
    for i in range(len(all_cond_entropy[0]) - 1):
        if len(special_points) + len(normal_points) == i: 
            if np.linalg.norm(all_cond_entropy[:,i] - all_cond_entropy[:,i+1]) < min_dist:
                special_points.append((all_cond_entropy[:,i], labels[i]))
                special_points.append((all_cond_entropy[:,i+1], labels[i+1]))
            else:
                normal_points.append((all_cond_entropy[:,i], labels[i]))
    
    if len(special_points) + len(normal_points) < len(all_cond_entropy[0]):
        normal_points.append((all_cond_entropy[:,-1], labels[-1]))
    
    
    # Plot non-overlapping points
    print("Annotate non-overlapping points")       
    for point, label in normal_points:
        plt.annotate(label, (point[0]+0.1, point[1]+0.1), annotation_clip=False)

    # Plot overlapping points
    print("Annotate overlapping points")
    texts = []
    for point, label in special_points:
        texts.append(plt.text(point[0], point[1], label))
    """

    # Plot points
    texts = []
    print("Annotate points")       
    for i, label in enumerate(labels):
        texts.append(plt.annotate(label, (all_cond_entropy[0,i], all_cond_entropy[1,i]), annotation_clip=False, fontsize=16))#, horizontalalignment="left")

    # Adjust text to avoid overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black'), min_arrow_len=6, expand=(1.5,1.7))#, time_lim=0.1)
    
    # Add labels and title
    plt.xlabel('H(Cluster|Class)', fontsize=16)
    plt.ylabel('H(Class|Cluster)', fontsize=16)
    plt.title(save_name)
    plt.grid(True)
    
    plt.savefig(f"vis/bias_abl_{dataset}_{model}.pdf")
    plt.close()
    

def plot_cond_entropy_train(all_cond_entropy, dataset, model, save_name, labels):
    """
    all_cond_entropy: list, contains the conditional entropy values for one model
    
    create a plot with points(x,y) which are the two conditional entropy values
    """
    # define plot
    plt.axis("scaled")
    plt.ylim(0,7)
    plt.xlim(0,4)
    plt.yticks(np.arange(0,8,1))
    plt.xticks(np.arange(0,5,1))

    #fit curve
    popt, pcov = curve_fit(func, all_cond_entropy[0], all_cond_entropy[1])
    print(popt, pcov)
    plot_points = np.arange(0, 5, 0.3)
    plt.plot(plot_points, func(plot_points, *popt),'--k', alpha=0.3)

    # Fit curve
    #ylog_data = np.log(all_cond_entropy[1])
    #curve_fit = np.polyfit(all_cond_entropy[0], ylog_data, 1)
    #print(curve_fit)
    #y_new = np.exp(curve_fit[0]) * np.exp(curve_fit[1]*all_cond_entropy[0])
    #plt.plot(all_cond_entropy[0], y_new, "--k", alpha=0.3)

    # plot points in all_cond_entropy (x,y)
    plt.plot(all_cond_entropy[0],all_cond_entropy[1], 'o',  zorder=10, clip_on=False)
    

    """
    # Check if any two points points are very near to each other
    min_dist = 0.5
    special_points = []
    normal_points = []
    for i in range(len(all_cond_entropy[0]) - 1):
        if len(special_points) + len(normal_points) == i: 
            if np.linalg.norm(all_cond_entropy[:,i] - all_cond_entropy[:,i+1]) < min_dist:
                special_points.append((all_cond_entropy[:,i], labels[i]))
                special_points.append((all_cond_entropy[:,i+1], labels[i+1]))
            else:
                normal_points.append((all_cond_entropy[:,i], labels[i]))
    
    if len(special_points) + len(normal_points) < len(all_cond_entropy[0]):
        normal_points.append((all_cond_entropy[:,-1], labels[-1]))
    
    
    # Plot non-overlapping points
    print("Annotate non-overlapping points")       
    for point, label in normal_points:
        plt.annotate(label, (point[0]+0.1, point[1]+0.1), annotation_clip=False)

    # Plot overlapping points
    print("Annotate overlapping points")
    texts = []
    for point, label in special_points:
        texts.append(plt.text(point[0], point[1], label))
    """

    # Plot points
    texts = []
    print("Annotate points")       
    for i, label in enumerate(labels):
        texts.append(plt.annotate(label, (all_cond_entropy[0,i], all_cond_entropy[1,i]), annotation_clip=False, fontsize=16))#, horizontalalignment="left")

    # Adjust text to avoid overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black'), min_arrow_len=6, expand=(1.5,1.7))#, time_lim=0.1)
    
    # Add labels and title
    plt.xlabel('H(Cluster|Class)', fontsize=16)
    plt.ylabel('H(Class|Cluster)', fontsize=16)
    plt.title(save_name)
    plt.grid(True)
    
    #plt.legend()

    plt.savefig(f"vis/bias_abl_{dataset}_{model}_train.png")
    plt.savefig(f"vis/bias_abl_{dataset}_{model}_train.pdf")
    plt.close()