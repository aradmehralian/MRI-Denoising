import numpy as np
import cv2

def adaptive_wiener_filter(image, kernel_size=5):
    """
    Apply Adaptive Wiener filter to an input image.

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - kernel_size: Size of the local window (odd number).

    Returns:
    - Filtered image.
    """
    image = image.astype(np.float64) # convert to float64 for better precision
    local_mean = cv2.blur(image, (kernel_size, kernel_size))
    local_var = cv2.blur(image ** 2, (kernel_size, kernel_size)) - local_mean ** 2
    noise_variance = np.mean(local_var)

    # Adaptive Wiener filter formula
    result = local_mean + (np.maximum(local_var - noise_variance, 0) / np.maximum(local_var, noise_variance)) * (image - local_mean)
    
    return result.astype(np.uint8)

