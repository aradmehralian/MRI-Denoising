import numpy as np

def calculate_median(array: np.ndarray) -> int:
    """
    Returns the median of 1-d array
    """
    sorted_array = np.sort(array) 
    median = sorted_array[len(array)//2]
    return median

def level_B(z_min: int, z_med: int, z_max: int, z_xy: int) -> int:
    if (z_min < z_xy < z_max):
        return z_xy
    else:
        return z_med
    
def level_A(z_min: int, z_med: int, z_max: int, z_xy: int, S_xy: int, S_max: int):
    if(z_min < z_med < z_max):
        return level_B(z_min, z_med, z_max, z_xy)
    else:
        S_xy += 2 # increase the size of S_xy to the next odd value.
        if (S_xy <= S_max): # repeat process
            return level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            return z_med
        
def adaptive_median_filter(image: np.ndarray, initial_window=3, max_window=11) -> np.ndarray:
    """
    Apply Adaptive Median Filter to an input image

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - initial_window: initial kernel size with default value 3.
    - max_window: maximum kernel size allowed with default value 11.

    Returns:
    - Filtered image.
    """
        
    h, w = image.shape 
    
    z_min, z_med, z_max, z_xy = 0, 0, 0, 0
    S_max = max_window
    S_xy = initial_window 
    
    output_image = image.copy()
    
    for row in range(S_xy, h-S_xy-1):
        for col in range(S_xy, w-S_xy-1):
            filter_window = image[row - S_xy : row + S_xy + 1, col - S_xy : col + S_xy + 1] # filter window
            target = filter_window.reshape(-1) # make it 1-dimensional
            z_min = np.min(target) # min of intensity values
            z_max = np.max(target) # max of intensity values
            z_xy = image[row, col] # current intensity
            z_med = calculate_median(target) #median of intensity values
            
            # Level A & B
            new_intensity = level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            output_image[row, col] = new_intensity
    return output_image