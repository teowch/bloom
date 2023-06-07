# ===============================================================================
# Author: Teodoro Valença de Souza Wacholski
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import numpy as np
import cv2

INPUT_IMAGE = "example.png"
BACKGROUND_THRESHOLD = 0.4

# INPUT_IMAGE = 'mine.png'
# BACKGROUND_THRESHOLD = 0.6

"""
Some steps are need to apply bloom
1. Isolate the light sources using a brigh-pass
2. Blur the light sources multiple times
3. Sum the original image to the blurred light sources using a Weighted Sum
    final = (0.9 * original) + (0.2 * light_source)

Here you will find two implementations of bloom.
They are boxBloom(using boxBlur) and gaussianBloom(using gaussianBlur).
Note that each gaussianBlur applied in the first solution is
replaced by three boxBlur in the second solution.
"""


def getLightSources(img, threshold):
    """
    Extract the light sources from the img based on their intensity.
    
    The intensity is obtained by converting the image to grayscale and comparing each pixel to the threshold.
    If the pixel value in the grayscale image is greater than or equal to threshold, it's considered
    a light source pixel. If it's not a light source pixel, its value will be set to zero.

    Params:
        img: input image
        threshold: threshold
    Returns:
        img_light_sources: an image with just the light sources
            This is what you will see in the "Light Sources" window when run the code
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_light_sources = np.copy(img)
    for y in range(len(img_gray)):
        for x in range(len(img_gray[0])):
            if img_gray[y, x] < threshold:
                img_light_sources[y, x] = 0
    return img_light_sources

def gaussianBloom(img, light_sources):
    """
    Apply bloom using gaussianBlur technique
    """

    backgroundBlurred = np.zeros_like(img)
    for i in range(1, 4):
        backgroundBlurred += cv2.GaussianBlur(light_sources, (75, 75), 4**i)
    imgReturn = 0.9 * img + 0.2 * backgroundBlurred
    cv2.imshow("Background Blur Gaussian", backgroundBlurred)
    return imgReturn


def boxBloom(img, light_sources):
    """
    Apply bloom using boxBlur technique
    """

    backgroundBlurred = np.zeros_like(light_sources)
    for i in range(1, 4):
        tmpBlurred = np.copy(light_sources)
        for _ in range(3):
            tmpBlurred = cv2.blur(tmpBlurred, (15 * i, 15 * i))
        backgroundBlurred += tmpBlurred
    imgReturn = 0.9 * img + 0.2 * backgroundBlurred
    cv2.imshow("Background Blur BoxBlur", backgroundBlurred)
    return imgReturn


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    cv2.imshow("No Effect", img)
    if img is None:
        print("Cannot open image")
        sys.exit()

    img = np.float32(img) / 255

    img_light_sources = getLightSources(img, BACKGROUND_THRESHOLD)
    
    cv2.imshow("Light Sources", img_light_sources)

    boxFinal = boxBloom(img, img_light_sources)
    gaussianFinal = gaussianBloom(img, img_light_sources)

    cv2.imshow("01 - Gaussian Bloom", gaussianFinal)
    cv2.imwrite("01 - Gaussian.png", gaussianFinal * 255)
    cv2.imshow("02 - Box Bloom", boxFinal)
    cv2.imwrite("02 - Box Bloom.png", boxFinal * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
