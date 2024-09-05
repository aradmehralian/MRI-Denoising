import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_gaussian_noise(image, std):
    """
    Adds Gaussian noise to a given image.

    Parameters:
    ----------
    image : numpy.ndarray
        The input image in grayscale format. It should be a 2D array representing pixel values.
    std : float
        The standard deviation of the Gaussian noise. Higher values of `std` result in more noise being added to the image.

    Returns:
    -------
    numpy.ndarray
        The noisy image with Gaussian noise added. The output image is of the same size and type as the input image.
     """
      
    noise = np.random.normal(0, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Adds salt-and-pepper noise to a given grayscale image.

    Parameters:
    ----------
    image : numpy.ndarray
        The input image in grayscale format. It should be a 2D array representing pixel values.
    salt_prob : float
        The probability of adding salt noise (white pixels). Should be a value between 0 and 1.
    pepper_prob : float
        The probability of adding pepper noise (black pixels). Should be a value between 0 and 1.

    Returns:
    -------
    numpy.ndarray
        The noisy image with salt-and-pepper noise added. The output image is of the same size and type as the input image.
    """
    noisy_image = image.copy()

    salt_mask = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_mask] = 255  # salt noise equals white (255)

    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0  # pepper noise equal black (0)

    return noisy_image


def imshow(image: np.ndarray, title: str, size: (int|float) = 5) -> None:
    """
    Shows image given its matrix along with title. Size of the image can be adjusted.

    Args:
        image (np.ndarray): Image matrix.
        title (str): Title of the image.
        size (int|float): Size of the image to be displayed.

    Returns:
        None
    """
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = h/w
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(image, cmap= 'gray')
    plt.title(title.title())
    plt.show()

def calculate_psnr(original: np.ndarray, filtered_img: np.ndarray):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between a noisy image and its noiseless counterpart.

    Parameters:
    ----------
    original : numpy.ndarray
        The original noiseless image. It should be a 2D or 3D array representing pixel values.
    noisy : numpy.ndarray
        The noisy image. It should be a 2D or 3D array of the same size and type as the original image.

    Returns:
    -------
    float
        The PSNR value, in decibels (dB). Higher values indicate better quality (less noise).
    """
    mse = np.mean((original - filtered_img) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if there is no noise (MSE is zero)
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr