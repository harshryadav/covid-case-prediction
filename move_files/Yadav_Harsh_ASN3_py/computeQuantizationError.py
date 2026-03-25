import numpy as np

def computeQuantizationError(origImg, quantizedImg):
    ''''
    Compute the Sum of Squared Differences (SSD) between the original and quantized images.

    SSD formula = sum((original_pixel - quantized_pixel) ** 2) for all pixels in the image
    '''
    original_image = np.asarray(origImg)
    quantized_image = np.asarray(quantizedImg)
    image_match = original_image.shape == quantized_image.shape
    if not image_match:
        print("Original and quantized images must have the same dimensions.")
        return None
    
    sum_squared_dff = np.sum((original_image.astype(np.float64) - quantized_image.astype(np.float64)) ** 2)
    return float(sum_squared_dff)