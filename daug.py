import numpy as np
"""
Data AUGmentation
--------------
Module for image data augmentation.
"""


def _force_RGB_array(img):
    """
    Ensures image is in the form of a (h, w, 3) array.

    Not intended for standalone use.

    Parameters
    ----------
    img - an RGB image or corresponding array, a grayscale image or corresponding array, or
    an image-like object convertible into numpy.array.

    Returns
    --------
    An array with shape (h, w, 3), where h is the height and w is the width of the original array
    """
    if not isinstance(img, np.array):
        img = np.array(img)
    if len(img.shape) == 3:
        return img
    elif len(img.shape) == 2:
        img = np.dstack([img, img, img])
        return img


def rot90(img, ccw=False):
    """
    Rotates image 90 degrees.

    Parameters
    ----------
    img - image to rotate

    ccw - if true, rotates the image counterclockwise. False by default.

    Returns
    -------
    Array corresponding to image, rotated 90 degrees.
    """
    img = _force_RGB_array(img)
    k = -1 if ccw else 1
    img = np.rot90(img, k)
    return img


def flipv(img):
    """
    Flips the image vertically (against the central horizontal axis)

    Parameters
    ----------
    img - image to flip

    Returns
    -------
    Array corresponding to the input image, flipped vertically.
    """
    img = _force_RGB_array(img)
    img = np.flipud(img)
    return img


def fliph(img):
    """
    Flips the image horizontally (against the central vertical axis)

    Parameters
    ----------
    img - image to flip

    Returns
    -------
    Array corresponding to the input image, flipped horizontally.
    """
    img = _force_RGB_array(img)
    img = np.fliplr(img)
    return img


def rolldown(img, px):
    """
    Shifts the image down by px pixels, with downmost rows coming around the top.
    Use negative px values for the same effect upwards.

    Parameters
    ---------
    img - image to roll

    px - number of pixels to shift by. Positive values result in a downward roll and
    negative values in an upward one.

    Returns
    -------
    Array corresponding to image, rolled down or up by px rows.
    """
    img = _force_RGB_array(img)
    px = int(px)
    img = np.roll(img, px, axis=0)
    return img
    

def rollright(img, px):
    """
    Shifts the image right by px pixels, with rightmost rows coming around the left.
    Use negative px values for the same effect left.

    Parameters
    ---------
    img - image to roll

    px - number of pixels to shift by. Positive values result in a roll to the right and
    negative values to the left.

    Returns
    -------
    Array corresponding to image, rolled right or left by px rows.
    """
    img = _force_RGB_array(img)
    px = int(px)
    img = np.roll(img, px, axis=1)
    return img


def zoom(img, fac=.5):
    """
    Zooms into the center of an image by fac.

    Parameters
    ----------
    img - image to zoom

    fac - percantage of the image to zoom; 2-long list or tuple of floats, or float. If float, image is zoomed by fac in each direction.
    If list or tuple, zoomed by fac[0]*height vertically and fac[1]*width horizontally. In both cases floats
    should be between 0 and 1.

    Returns
    -------
    Array corresponding to the central (height*fac, width*fac) region of the image.
    """
    img = _force_RGB_array(img)
    if not isinstance(fac, tuple) and not isinstance(fac, list):
        fac = (fac, fac)
    if fac[0] > 1:
        fac = (1, fac[1])
    if fac[1] > 1:
        fac = (fac[0], 1)
    if fac[0] <= 0:
        fac = (.5, fac[1])
    if fac[1] <= 0:
        fac = (fac[0], .5)
    h, w, _ = img.shape
    bounds_h = [int(h/2-h*(fac[0]/2.)), int(h/2+h*(fac[0]/2.))]
    bounds_w = [int(w/2-w*(fac[1]/2.)), int(w/2+w*(fac[1]/2.))]
    img = img[bounds_h[0]:bounds_h[1], bounds_w[0]:bounds_w[1], :]
    return img
    

def crop(img, bounds):
    """
    Crops the image within bounds.

    Parameters
    ----------
    img - image to crop

    bounds - crop start and stop pixel indices, in the form: [[y_start, x_start], [y_stop, x_stop]]
    Values can't exceed image size.

    Returns 
    ------
    Array corresponding to the cropped image region
    """
    img = _force_RGB_array(img)
    img = img[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], :]
    return img


def noisy(img, amt):
    """
    Adds noise to the image.

    Parameters
    ----------
    img - image to augment

    amt - amount of noise to use. Float between 0 and 1, 1 being maximum image brightness.

    Returns
    -------
    Input image in the form of a 3-dimensional array, with added noise.
    """
    img = _force_RGB_array(img)
    noise = np.random.uniform(size=img.shape) * 255
    img = img + noise*amt
    return img


def mirrorleft(img):
    """
    Mirrors the left half of the image to the right.

    Parameters 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the vertical axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.fliplr(img)
    bnd = img.shape[1] // 2
    img[:, bnd:, :] = flipclone[:, bnd:, :]
    return img


def mirrorright(img):
    """
    Mirrors the right half of the image to the left.

    Parameters 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the vertical axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.fliplr(img)
    bnd = img.shape[1] // 2
    img[:, :bnd, :] = flipclone[:, :bnd, :]
    return img


def mirrorup(img):
    """
    Mirrors the upper half of the image to the bottom.

    Parameters 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the hoizontal axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.flipud(img)
    bnd = img.shape[0] // 2
    img[bnd:, :, :] = flipclone[bnd:, :, :]
    return img


def mirrordown(img):
    """
    Mirrors the bottom half of the image to the top.

    Parameters 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the hoizontal axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.flipud(img)
    bnd = img.shape[0] // 2
    img[:bnd, :, :] = flipclone[:bnd, :, :]
    return img


def shiftdown(img, px):
    """
    Shifts the image down by px, with the gap at the top filled by stretching the topmost row of pixels.

    Parameteres
    ----------
    img - image to shift

    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = rolldown(img, px)
    if px > 0:
        for i in range(px):
            img[i, :, :] = img[px, :, :]
    return img


def shiftup(img, px):
    """
    Shifts the image up by px, with the gap at the bottom filled by stretching the bottom row of pixels.

    Parameteres
    ----------
    img - image to shift

    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = rolldown(img, -px)
    if px > 0:
        for i in range(px):
            img[img.shape[0] - i - 1, :, :] = img[img.shape[0] - px, :, :]
    return img


def shiftleft(img, px):
    """
    Shifts the image left by px, with the gap at the right filled by stretching the rightmost row of pixels.

    Parameteres
    ----------
    img - image to shift

    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = rollright(img, -px)
    if px > 0:
        for i in range(px):
            img[:, img.shape[1] - i - 1, :] = img[:, img.shape[1] - px, :]
    return img


def shiftright(img, px):
    """
    Shifts the image right by px, with the gap at the left filled by stretching the leftmost row of pixels.

    Parameteres
    ----------
    img - image to shift

    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = rollright(img, px)
    if px > 0:
        for i in range(px):
            img[:, i, :] = img[:, px, :]
    return img













