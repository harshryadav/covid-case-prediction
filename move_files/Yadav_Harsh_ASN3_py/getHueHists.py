import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv

def getHueHists(im, k):
    '''
    Computes two hue histograms:
        - One with equally spaced bins
        - One using cluster-based bins
    '''
    image = np.asarray(im)

    TWO_PI = 2.0 * np.pi

    if image.ndim != 3 or image.shape[2] != 3:
        print("Input image must be RGB with 3 channels")
        return None, None, None, None

    is_integer = np.issubdtype(image.dtype, np.integer)
    if is_integer or image.max() > 1.0:
        image_rgb = image.astype(np.float64) / 255.0
    else:
        image_rgb = image.astype(np.float64)
    
    hue_channel = rgb_to_hsv(image_rgb)[:, :, 0].ravel()

    hist, bins = np.histogram(hue_channel, bins=k, range=(0.0, 1.0))

    hue_theta = TWO_PI * hue_channel
    feats = np.column_stack([np.cos(hue_theta), np.sin(hue_theta)])

    # K-means clustering on the Hue channel
    k_means = KMeans(n_clusters=k, n_init=5)
    k_means_labels = k_means.fit(feats).predict(feats)
    histogram_cluster = np.bincount(k_means_labels, minlength=k).astype(float)

    cluster_x = k_means.cluster_centers_[:, 0]
    cluster_y = k_means.cluster_centers_[:, 1]
    cluster_hue = (np.arctan2(cluster_y, cluster_x) / TWO_PI) % 1.0

    return hist, bins, histogram_cluster, cluster_hue

    
