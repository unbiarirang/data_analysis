# !/usr/bin/env python3
# encoding=utf-8

from sklearn.cluster import DBSCAN
import numpy as np
import os
from utils.preprocess import load_data
from utils.metrics import silhouette_score_
from utils.visualize import visualize, MDS_visualize

output_dir = "../output"
data_dir = '../data'

def dbscan(raw_data, eps = 5):
    x = np.array(raw_data)
    clusters = DBSCAN(eps = eps, min_samples=5).fit(x)
    # print(km.cluster_centers_)
    # print(km.labels_)

    distance = 1 - cosine_similarity(x)
    mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 1)
    pos = mds.fit_transform(distance)
    xs, ys = pos[:, 0], pos[:, 1]

    #for x_, y_ in zip(xs, ys):
    #    plt.scatter(x_, y_)
    #plt.title('MDS output')
    #plt.savefig(os.path.join(output_dir, 'MDS.png'))

    
    df = pd.DataFrame(dict(label = clusters.labels_, data=raw_data, x=xs, y=ys))

    label_color_map = {
                        -1: 'black',
                        0: 'red',
                        1: 'blue',
                        2: 'green',
                        3: 'pink',
                        4: 'purple',
                        5: 'yellow',
                        6: 'orange',
                        7: 'grey',
                        8: 'purple',
                        9: 'violet',
                        10: '#FF0000',
                        11: '#FF0000',
                        12: '#FF0000',
                        13: '#FF0000',
                        }

    fig, ax = plt.subplots(figsize=(17, 9))

    for index, row in df.iterrows():
        cluster = row['label']
        # print(cluster)
        label_color = label_color_map[row['label']]
        # label_text = row['data']
        ax.plot(row['x'], row['y'], marker='o', ms=12, c=label_color)
        # row = str(cluster) + ',' + label_text + '\n'
        # csv.write(row)

    # ax.legend(numpoints=1)
    # for i in range(len(df)):
    #    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['label'], size=8)

    plt.title('DBSCAN Clustering')
    n_clusters = set(clusters.labels_)
    n_clusters.add(-1)
    n_clusters.remove(-1)
    n_clusters = len(n_clusters)

    plt.savefig(os.path.join(output_dir, 'DBSCAN_' + str(n_clusters) + '.png'))



if __name__ == "__main__":
    data = read_data(os.path.join(data_dir, 'cluster_data.txt'))
    dbscan(data, 6)
