# !/usr/bin/env python3
# encoding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
import os


label_color_map = {8: '#CC3333', # red
                1: '#99CC33', # light green
                2: '#FFCC33', # yellow
                3: '#3399CC', # blue
                # 3: '#3366CC', # blue
                4: '#CCCCFF', # purple
                5: '#999999', # grey
                6: '#FF9933', # orange
                7: '#6666CC', # blue & purple
                0: '#E03636',
                -1: 'black'

                }

def MDS_visualize(data, path):
    print('dimension reduction...')
    distance = 1 - cosine_similarity(data)
    mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 1)
    pos = mds.fit_transform(distance)
    # print(pos.shape)
    xs, ys = pos[:, 0], pos[:, 1]

    for x_, y_ in zip(xs, ys):
        plt.scatter(x_, y_)
    plt.title('MDS output')
    plt.savefig(path)
    print('dimension reduction results saved in %s' %path)

def TSNE_visualize(data, path, random_state):
    print('dimension reduction...')
    distance = 1 - cosine_similarity(data)
    tsne = TSNE(n_components = 2, random_state = random_state)
    pos = tsne.fit_transform(distance)
    # print(pos.shape)
    xs, ys = pos[:, 0], pos[:, 1]

    for x_, y_ in zip(xs, ys):
        plt.scatter(x_, y_)
    plt.title('TSNE output')
    plt.savefig(path)
    print('dimension reduction results saved in %s' %path)


def visualize(raw_data, x, y, n_clusters, method, path, reduction_method, random_state, **kargs):
    print('visualize results...')
    distance = 1 - cosine_similarity(x)
    if reduction_method.upper() == 'MDS':
        res = MDS(n_components=2, dissimilarity="precomputed", random_state = random_state)
    elif reduction_method.upper() == 'TSNE':
        res = TSNE(n_components=2, random_state = random_state, perplexity=30)
    pos = res.fit_transform(distance)
    # print(pos.shape)
    xs, ys = pos[:, 0], pos[:, 1]

    df = pd.DataFrame(dict(label = y, data=raw_data, x=xs, y=ys))
    fig, ax = plt.subplots(figsize=(17, 9))
    for index, row in df.iterrows():
        cluster = row['label']
        label_color = label_color_map[row['label']]
        # label_text = row['data']
        ax.plot(row['x'], row['y'], marker='o', ms=12, c=label_color)
        # row = str(cluster) + ',' + label_text + '\n'
        # csv.write(row)
        # ax.legend(numpoints=1)
        # for i in range(len(df)):
        #    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['label'], size=8)
    
    plt.title('%s Clustering for %d classes' %(method, n_clusters))
    ax = fig.add_axes([0.78, 0.1, 0.2, 0.2])                
    #在（0.5，0）到（0.5，1）区域写入标注，前两个参数是相对位置
    ax.text(0.1,0.35, U'intra_distance: %f' %kargs['intra'], transform=ax.transAxes,fontdict = {'size': 8, 'color': 'black'})
    ax.text(0.1,0.25, U'inter_distance: %f' %kargs['inter'], transform=ax.transAxes,fontdict = {'size': 8, 'color': 'black'})
    ax.text(0.1,0.15, U'silhouette score: %f' %kargs['score'], transform=ax.transAxes,fontdict = {'size': 8, 'color': 'black'})
    ax.set_axis_off()

    plt.savefig(path)
    print('Visualized results saved in %s' %path)

def visualize_proc(X, y, method, path):

    print('visualize hierarchical tree...')   
    distance = 1 - cosine_similarity(X)

    # Ward’s method produces a hierarchy of clusterings
    linkage_matrix = ward(distance)
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="top", labels=y)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.title('Hierarchical Method Process')
    plt.savefig(path)
    print('Visualized results saved in %s' %path)