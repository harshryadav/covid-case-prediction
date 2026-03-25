import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.cluster import KMeans


def quantizeHSV(origImg, k):
    '''
        Converts an RGB image to HSV and quantizes only the Hue channel.
        HSV stands for Hue, Saturation, and Value
    '''

    image = np.asarray(origImg)

    TWO_PI = 2.0 * np.pi

    if image.ndim != 3 or image.shape[2] != 3:
        print("Input image must be RGB with 3 channels")
    
    is_integer = np.issubdtype(image.dtype, np.integer)

    if is_integer or image.max() > 1.0:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
    
    # convert RGB to HSV
    hsv_channel = rgb_to_hsv(image_float)
    h, w = hsv_channel.shape[0:2]
    hue_channel = hsv_channel[:, :, 0].reshape(-1, 1)
    saturation_channel = hsv_channel[:, :, 1].reshape(-1, 1)
    value_channel = hsv_channel[:, :, 2].reshape(-1, 1)

    # Convert hue to radians and multiply by 2pi for cartesian coordinates for clustering
    hue_theta = TWO_PI * hue_channel.ravel()
    feats = np.column_stack([np.cos(hue_theta), np.sin(hue_theta)])

    # K-means clustering on the Hue channel
    k_means = KMeans(n_clusters=k, n_init=5)
    k_means_labels = k_means.fit(feats).predict(feats)

    # Get the cluster centers and labels
    cluster_x = k_means.cluster_centers_[:, 0]
    cluster_y = k_means.cluster_centers_[:, 1]
    cluster_hue = (np.arctan2(cluster_y, cluster_x) / TWO_PI) % 1.0 

    # map each pixel to the nearest cluster center
    hue_channel_quantized = cluster_hue[k_means_labels].reshape(h, w)
    saturation_channel_reshaped = saturation_channel.reshape(h, w)
    value_channel_reshaped = value_channel.reshape(h, w)
    updated_hsv = np.stack([hue_channel_quantized, saturation_channel_reshaped, value_channel_reshaped], axis=2)
    new_rgb = hsv_to_rgb(updated_hsv)
    quantized_image = np.clip(np.round(new_rgb * 255.0), 0, 255).astype(np.uint8)

    return quantized_image, cluster_hue

