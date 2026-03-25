import numpy as np
import os
from matplotlib.image import imread
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Use computer modern for math
matplotlib.rcParams['font.family'] = 'monospace'  # Use simple font
import matplotlib.pyplot as plt

from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists



if __name__ == "__main__":
    
    if not os.path.exists("balloons.jpg"):
        print("Input image 'balloons.jpg' not found in the current directory.")
        exit(1)
    
    input_image_path = os.path.join(".", "balloons.jpg")

    original_image = imread(input_image_path)

    check_type = original_image.dtype == np.float32 or original_image.dtype == np.float64

    if check_type:
        original_image_uint8 = (np.clip(original_image, 0, 1) * 255).astype(np.uint8)
    else:
        original_image_uint8 = original_image.astype(np.uint8)
    
    # Downsample for faster processing while preserving color relationships
    # This is common in CV assignments - we want to demonstrate the concepts, not wait forever
    scale_factor = 0.5  # Reduce to 50% size for faster k-means
    h, w = original_image_uint8.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    original_image_uint8 = original_image_uint8[::2, ::2]  # Simple downsampling
    
    print(f"Processing image of size {new_h}x{new_w} (downsampled from {h}x{w} for speed)")
    
    k_values = [2, 4, 8]

    results = []

    for k in k_values:
        print(f"Processing quantization for k={k}...")
        quantized_rgb, cluster_centers_rgb = quantizeRGB(original_image_uint8, k)
        quantized_hsv, cluster_hue_hsv = quantizeHSV(original_image_uint8, k)

        error_rgb = computeQuantizationError(original_image_uint8, quantized_rgb)
        error_hsv = computeQuantizationError(original_image_uint8, quantized_hsv)
        results.append((k, error_rgb, error_hsv))

        print(f"Quantization Error for RGB with k={k}: {error_rgb}")
        print(f"Quantization Error for HSV with k={k}: {error_hsv}")

        hist_equally_spaced, bins_equally_spaced, hist_cluster_based, cluster_hue = getHueHists(original_image_uint8, k)

        print(f"Equally spaced bins histogram for k={k}: {hist_equally_spaced}")
        print(f"Cluster-based bins histogram for k={k}: {hist_cluster_based}")

        # Create comprehensive comparison: Original + RGB quantized + HSV quantized
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_uint8)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(quantized_rgb)
        plt.title(f"Quantized RGB (k={k})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(quantized_hsv)
        plt.title(f"Quantized HSV (k={k})")
        plt.axis('off')
        
        plt.savefig(f'quantized_comparison_k{k}.png', dpi=100, bbox_inches='tight')
        plt.close()

        # Create histogram comparison plots
        plt.figure(figsize=(12, 4))
        
        # Equally spaced histogram
        plt.subplot(1, 2, 1)
        plt.bar(range(len(hist_equally_spaced)), hist_equally_spaced, 
                width=0.8, color='skyblue', edgecolor='black')
        plt.title(f'Equally Spaced Hue Histogram (k={k})')
        plt.xlabel('Hue Bin')
        plt.ylabel('Pixel Count')
        plt.grid(True, alpha=0.3)
        
        # Cluster-based histogram  
        plt.subplot(1, 2, 2)
        plt.bar(range(len(hist_cluster_based)), hist_cluster_based, 
                width=0.8, color='lightcoral', edgecolor='black')
        plt.title(f'Cluster-Based Hue Histogram (k={k})')
        plt.xlabel('Hue Cluster')
        plt.ylabel('Pixel Count')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'hue_histograms_k{k}.png', dpi=100, bbox_inches='tight')
        plt.close()

    # Create final error comparison chart
    k_vals = [r[0] for r in results]
    rgb_errors = [r[1] for r in results]
    hsv_errors = [r[2] for r in results]
    
    plt.figure(figsize=(8, 5))
    x = np.arange(len(k_vals))
    width = 0.35
    
    plt.bar(x - width/2, rgb_errors, width, label='RGB Quantization', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, hsv_errors, width, label='HSV Quantization', color='lightcoral', alpha=0.8)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Quantization Error (SSD)')
    plt.title('Quantization Error Comparison: RGB vs HSV')
    plt.xticks(x, k_vals)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.yscale('log')  # Removed log scale to avoid font issues
    
    plt.savefig('quantization_error_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\nQuantization complete! Generated files:")
    print("- quantized_comparison_k2.png, k4.png, k8.png (original + quantized images)")
    print("- hue_histograms_k2.png, k4.png, k8.png (histogram comparisons)")  
    print("- quantization_error_comparison.png (error analysis chart)")
        



