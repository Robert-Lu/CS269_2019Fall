import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from chan_vese import ChanVeseModel
from scipy.misc import imresize
from time import sleep

from plt_helper import *

from util import rgb2gray, create_shape_on_image, create_mask_from_shape


def main():
    print("Test")

    # import image
    save = "example42"
    save = None
    img = plt.imread("./example3.jpg")

    size = img.shape
    if size[0] > 250:
        h = size[0]
        w = size[1]
        while max(h, w) > 250:
            h = h // 2
            w = w // 2
        img = imresize(img, (h, w))

    # convert to grayscale if image has 3rd rank (RGBA channels)
    if len(img.shape) == 3:
        img = rgb2gray(img)

    # normalize
    img = img.astype(float) / np.max(img)
    width = img.shape[1]
    height = img.shape[0]

    ## Main Chan-Vese process
    # get a shape from mouse clicking on GUI
    shapes = create_shape_on_image(img, max_shape_num=1)
    shape = shapes[0]
    # create a mask from the shape
    mask = create_mask_from_shape(width, height, shape)
    # show the mask
    imagesc(mask.astype(int), colorbar=False, title="The Mask")
    
    # create a Chan-Vese Model
    model = ChanVeseModel(img, mask, check_same_every_n_iter=50)
    i = 0
    model.draw(save=save)
    model.draw(save=save)
    while not model.done():
        model.iterate(_lambda1=1, _lambda2=1, _mu=.1, _nu=.02, _dt=.5)
        if i % 10 == 9:
            model.draw(save=save)
        i += 1
    model.draw(block=True)



if __name__ == "__main__":
    main()
