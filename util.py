import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt 

from plt_helper import *


def build_dist_map_from_mask(mask, epsilon=0):
    bwdist = lambda im: distance_transform_edt(np.logical_not(im))
    bw = mask
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    d = signed_dist.astype(np.float64)

    d += epsilon
    while np.count_nonzero(d < 0) < 5:
        d -= 1
    return d


class LineBuilder:
    """
        https://stackoverflow.com/questions/37363755/python-mouse-click-coordinates-as-simply-as-possible
        Altered
    """

    def __init__(self, line, ax, fig, color, max_shape_num = 1):
        self.line = line
        self.ax = ax
        self.color = color
        self.fig = fig
        self.max_shape_num = max_shape_num
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}
        self.precision = 10

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        if self.counter == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        if np.abs(event.xdata - self.xs[0]) <= self.precision and np.abs(event.ydata - self.ys[0]) <= self.precision and self.counter != 0:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.ax.scatter(self.xs, self.ys, s=120, color=self.color)
            self.ax.scatter(self.xs[0], self.ys[0], s=80, color='blue')
            self.ax.plot(self.xs, self.ys, color=self.color)
            self.line.figure.canvas.draw()
            self.shape[self.shape_counter] = [self.xs, self.ys]
            self.shape_counter = self.shape_counter + 1
            self.xs = []
            self.ys = []
            self.counter = 0
        else:
            if self.counter != 0:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
            self.ax.scatter(self.xs, self.ys, s=120, color=self.color)
            self.ax.plot(self.xs, self.ys, color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1

        if self.shape_counter >= self.max_shape_num:
            plt.close(self.fig)


def create_shape_on_image(data, max_shape_num = 1):
    """
        https://stackoverflow.com/questions/37363755/python-mouse-click-coordinates-as-simply-as-possible
    """
    
    def change_shapes(shapes):
        new_shapes = {}
        for i in range(len(shapes)):
            l = len(shapes[i][1])
            new_shapes[i] = np.zeros((l, 2), dtype='int')
            for j in range(l):
                new_shapes[i][j, 0] = shapes[i][0][j]
                new_shapes[i][j, 1] = shapes[i][1][j]
        return new_shapes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(
        'Please click draw a shape (10 pixel precision to close the shape)')
    line = ax.imshow(data, cmap="gray")
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    linebuilder = LineBuilder(line, ax, fig, 'red', max_shape_num=1)
    plt.gca().invert_yaxis()
    plt.show()
    new_shapes = change_shapes(linebuilder.shape)
    return new_shapes

def create_mask_from_shape(width, height, shape):
    """
        https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    """

    nx, ny = width, height
    poly_verts = shape

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))

    return grid


def rgb2gray(rgb):
    """
        From: 
        https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
