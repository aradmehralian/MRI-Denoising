import numpy as np
import cv2
from sklearn.decomposition import DictionaryLearning
from scipy.ndimage import uniform_filter

def bilateral_filter_decomposition(image, d=9, sigma_color=75, sigma_space=75):
    """
    Decompose the image using bilateral filtering into an edge layer and a residual layer.
    
    Parameters:
    - image: Input grayscale image (2D numpy array).
    - d: Diameter of each pixel neighborhood used in the bilateral filter.
    - sigma_color: Filter sigma in the color space.
    - sigma_space: Filter sigma in the coordinate space.
    
    Returns:
    - edge_layer: Edge-preserved layer of the image.
    - residual_layer: Residual layer containing fine details and noise.
    """
    # Apply bilateral filter to get the edge-preserved layer
    edge_layer = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    # Calculate the residual layer (original - edge layer)
    residual_layer = image.astype(np.float64) - edge_layer.astype(np.float64)
    
    return edge_layer, residual_layer


def k_svd_denoising(residual_layer, patch_size=16, n_components=50, max_iter=5):
    """
    Apply the K-SVD algorithm to denoise the residual layer.
    
    Parameters:
    - residual_layer: Residual layer of the image.
    - patch_size: Size of the patches to extract from the residual layer.
    - n_components: Number of dictionary components (atoms).
    - max_iter: Number of iterations for dictionary learning.
    
    Returns:
    - denoised_residual_layer: Denoised version of the residual layer.
    """
    # Extract patches from the residual layer
    patches = extract_patches(residual_layer, patch_size)
    
    # Apply K-SVD dictionary learning
    dico = DictionaryLearning(n_components=n_components, max_iter=max_iter, transform_algorithm='omp')
    dico.fit(patches)
    dictionary = dico.components_
    
    # Reconstruct the denoised patches
    denoised_patches = dico.transform(patches).dot(dictionary)
    
    # Reconstruct the residual layer from the denoised patches
    denoised_residual_layer = reconstruct_image_from_patches(denoised_patches, residual_layer.shape, patch_size)
    
    return denoised_residual_layer


def extract_patches(image, patch_size):
    """
    Extract overlapping patches from the image.
    
    Parameters:
    - image: Input image.
    - patch_size: Size of each patch (square).
    
    Returns:
    - patches: 2D array where each row is a flattened patch.
    """
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1):
        for j in range(0, image.shape[1] - patch_size + 1):
            patch = image[i:i + patch_size, j:j + patch_size].flatten()
            patches.append(patch)
    return np.array(patches)


def reconstruct_image_from_patches(patches, image_shape, patch_size):
    """
    Reconstruct the image from denoised patches.
    
    Parameters:
    - patches: 2D array of denoised patches.
    - image_shape: Shape of the original image.
    - patch_size: Size of each patch (square).
    
    Returns:
    - reconstructed_image: Reconstructed image from patches.
    """
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    count = np.zeros(image_shape, dtype=np.float64)
    patch_idx = 0
    for i in range(0, image_shape[0] - patch_size + 1):
        for j in range(0, image_shape[1] - patch_size + 1):
            reconstructed_image[i:i + patch_size, j:j + patch_size] += patches[patch_idx].reshape((patch_size, patch_size))
            count[i:i + patch_size, j:j + patch_size] += 1
            patch_idx += 1
    reconstructed_image /= count
    return reconstructed_image


def merge_layers(edge_layer, denoised_residual_layer):
    """
    Merge the edge-preserved layer with the denoised residual layer.
    
    Parameters:
    - edge_layer: Edge-preserved layer from bilateral filter.
    - denoised_residual_layer: Denoised residual layer from K-SVD.
    
    Returns:
    - final_image: Reconstructed denoised image.
    """
    final_image = edge_layer + denoised_residual_layer
    return np.clip(final_image, 0, 255).astype(np.uint8)


