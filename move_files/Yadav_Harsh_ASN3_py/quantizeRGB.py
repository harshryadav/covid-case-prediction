import numpy as np
from sklearn.cluster import KMeans


def quantizeRGB(origImg, k):
    '''
        Quantizes an RGB image into k clusters using k-means clustering.
    '''
    image = np.asarray(origImg)

    if image.ndim != 3 or image.shape[2] != 3:
        print("Input image must be RGB with 3 channels")

    is_integer = np.issubdtype(image.dtype, np.integer) 
    
    if is_integer or image.max() > 1.0:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
    
    # shapes of the image
    h, w, c = image_float.shape

    image_pixels = image_float.reshape(-1, 3)

    # K-means clustering
    k_means = KMeans(n_clusters=k, n_init=5)
    k_means_labels = k_means.fit(image_pixels).predict(image_pixels)

    # Get the cluster centers and labels
    cluster_centers = np.clip(k_means.cluster_centers_, 0, 1)

    # map each pixel to the nearest cluster center
    mapped_image = cluster_centers[k_means_labels].reshape(h, w, c)
    quantized_image = (mapped_image * 255.0).astype(np.uint8)
    rounded_image = np.round(quantized_image / 255.0 * 255).astype(np.uint8)

    return rounded_image, cluster_centers
