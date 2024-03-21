import numpy as np
import cv2
from sklearn.cluster import KMeans
import math


def crop_circle(image, center, radius):
    """
    Crops and applies a circular mask to a specified region in an input image.
    The function creates a mask that highlights a circular region based on the
    provided center and radius, then applies this mask to the input image to
    isolate the region of interest. The result is a cropped image containing
    only the circular area, with all pixels outside this area set to black.

    Parameters:
    - image (numpy.ndarray): The input image from which a circular region is to be cropped.
    - center (tuple): A tuple (x, y) representing the center of the circle.
    - radius (int): The radius of the circle.

    Returns:
    - numpy.ndarray: The cropped image containing only the circular region of interest,
                     with all other areas masked out.
    """
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y = center
    x1, y1 = max(x - radius, 0), max(y - radius, 0)
    x2, y2 = min(x + radius, image.shape[1]), min(y + radius, image.shape[0])

    cropped_image = masked_image[y1:y2, x1:x2]

    return cropped_image


def find_combined_dominant_color(image, center, radius, n_clusters=3):
    """
    Determines the dominant color within a specified circular region of an input image.
    This is achieved by first applying a mask to isolate the circular region and then
    performing k-means clustering on the pixels within this region to find clusters of
    similar colors. The dominant color is calculated as a weighted average of these clusters,
    where the weight is the proportion of pixels in each cluster.

    Parameters:
    - image (numpy.ndarray): The input image containing the region of interest.
    - center (tuple): A tuple (x, y) representing the center of the circle.
    - radius (int): The radius of the circular region.
    - n_clusters (int, optional): The number of clusters to use for k-means clustering. Defaults to 3.

    Returns:
    - tuple: A tuple (R, G, B) representing the dominant color in the RGB color space.
    """

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    masked_pixels = masked_image[mask == 255]

    masked_pixels = np.float32(masked_pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(masked_pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))

    weights = hist / np.sum(hist)

    weighted_sum = sum(weight * center for weight, center in zip(weights, centers))
    dominant_color = np.uint8(weighted_sum)
    dominant_color = tuple([int(c) for c in dominant_color])

    return dominant_color


def get_dynamic_circle_diameter(w, h):
    """
    Calculates a dynamic diameter for a circle based on the dimensions of a detected object.
    The function scales the circle's diameter in relation to the object's size, ensuring
    that the circle fits well within the object. This approach allows for a more adaptive
    representation of objects of various sizes, enhancing the accuracy of subsequent
    color analysis within these circular regions.

    Parameters:
    - w (int): The width of the detected object.
    - h (int): The height of the detected object.

    Returns:
    - int: The calculated diameter of the circle.
    """
    object_size = max(w, h)
    circle_diameter_factor = 0.3 + (min(max(object_size / w, 0.1), 0.5) - 0.1) * 0.5
    radius = int(min(w, h) * circle_diameter_factor / 2)
    return radius
