import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

# copy from https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return (scaled_inertia, kmeans.labels_)

def chooseBestKforKMeansParallel(scaled_data, k_range):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns 
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    best_labels: numpy Array
        predicted labels with chosen best k
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    
    ans = Parallel(n_jobs=-1, verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in k_range)
    scaled_inertia = list(zip(k_range, [x[0] for x in ans]))
    results = pd.DataFrame(scaled_inertia, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    best_labels = ans[best_k][1]
    return best_k, best_labels, results