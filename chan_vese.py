import numpy as np
import matplotlib.pyplot as plt

from util import build_dist_map_from_mask
from plt_helper import *

eps = np.finfo(float).eps


class ChanVeseModel(object):
    """ChanVeseModel"""
    
    def __init__(self, img, mask, max_iter=1000, check_same_every_n_iter=10, check_same_tolerence=0.001):
        self.iter_num = 0;
        self.check_same_every_n_iter = check_same_every_n_iter
        self.max_iter = max_iter
        self.check_same_tolerence = check_same_tolerence
        self.img = img
        self.mask = mask
        self.last_mask = None
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.phi = build_dist_map_from_mask(mask)
        self.last_phi = self.phi.copy()
        self.fig = None

    def iterate(self, _lambda1=1, _lambda2=1, _mu=1, _nu=0, _dt=0.5):
        print("Iteration ", self.iter_num)

        img = self.img
        phi = self.phi

        # Grab the curve's narrow band indices
        interior_idx = np.flatnonzero(np.logical_and(phi <= 0, phi >= -3))
        exterior_idx = np.flatnonzero(np.logical_and(phi <= 3, phi > 0))
        idx = np.concatenate((interior_idx, exterior_idx))

        # Calculate c_1(\phi^n) and c_2(\phi^n) in paper
        interior_area = np.flatnonzero(phi <= 0) # interior points
        exterior_area = np.flatnonzero(phi > 0)  # exterior points
        c_in  = np.sum(img.flat[interior_area]) / (len(interior_area) + eps)  # interior mean
        c_out = np.sum(img.flat[exterior_area]) / (len(exterior_area) + eps)  # exterior mean
        
        # Calculate image force term (\Int{(u_0 - c_1or2)^2})
        # force_img_in = (img.flat[interior_idx] - c_in) ** 2 
        # force_img_out = - (img.flat[exterior_idx] - c_out) ** 2
        # force_img = np.concatenate((force_img_in, force_img_out))
        force_img = _lambda1 * (img.flat[idx] - c_in)**2 - _lambda2 * (img.flat[idx] - c_out)**2

        # Calculate curvature of \phi
        curvature = get_curvature(phi, idx)

        dphidt = force_img / np.max(np.abs(force_img)) + _mu * curvature + _nu

        delta_t = _dt

        self.last_phi = self.phi.copy()
        self.phi.flat[idx] += delta_t * dphidt
        self.phi = sussman(self.phi, delta_t)

        if self.iter_num != 0 and self.iter_num % self.check_same_every_n_iter == 0:
            self.last_mask = self.mask.copy()
            self.mask = np.less(self.phi, 0)

        self.iter_num += 1;


    def draw(self, block=False, save=None):
        title = "Iteration %d" % self.iter_num
        if self.fig is None:
            # self.fig = show_chan_vese(self.img, self.phi, title=title, block=block, cmap="gray", zls_color=[1,0,0])
            self.fig = show_chan_vese(self.img, self.phi, fig_handle=None, title=title, block=block, cmap="gray", zls_color=[1,0,0])
        else:
            self.fig = show_chan_vese(self.img, self.phi, fig_handle=self.fig, title=title, block=block, cmap="gray", zls_color=[1,0,0])
        if save is not None:
            plt.savefig('./images/%s%4d.png' % (save, self.iter_num))

    def done(self):
        if self.iter_num >= self.max_iter:
            return True
        if self.last_mask is None:
            return False

        diff = np.sum(np.not_equal(self.mask, self.last_mask).astype(int))
        total = self.width * self.height
        return (diff / total) < self.check_same_tolerence



# Helpers
# some code are from https://github.com/kevin-keraudren/chanvese

# Compute curvature along SDF
def get_curvature(phi, idx):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                     phi.flat[iddr] + phi.flat[idul])
    phi_x2 = phi_x**2
    phi_y2 = phi_y**2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + eps) ** 1.5) * (phi_x2 + phi_y2) ** 0.5

    return curvature


# Level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - np.roll(D, 1, axis=1)
    b = np.roll(D, -1, axis=1) - D
    c = D - np.roll(D, -1, axis=0)
    d = np.roll(D, 1, axis=0) - D

    a_p = np.clip(a, 0, np.inf)
    a_n = np.clip(a, -np.inf, 0)
    b_p = np.clip(b, 0, np.inf)
    b_n = np.clip(b, -np.inf, 0)
    c_p = np.clip(c, 0, np.inf)
    c_n = np.clip(c, -np.inf, 0)
    d_p = np.clip(d, 0, np.inf)
    d_n = np.clip(d, -np.inf, 0)

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros_like(D)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_p.flat[D_pos_ind]**2], [b_n.flat[D_pos_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_p.flat[D_pos_ind]**2], [d_n.flat[D_pos_ind]**2])), axis=0)) - 1
    dD.flat[D_neg_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_n.flat[D_neg_ind]**2], [b_p.flat[D_neg_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_n.flat[D_neg_ind]**2], [d_p.flat[D_neg_ind]**2])), axis=0)) - 1
    dD[ 0,:] = 0
    dD[-1,:] = 0
    dD[:, 0] = 0
    dD[:,-1] = 0

    D -= dt * sussman_sign(D) * dD
    return D


def sussman_sign(D):
    return D / np.sqrt(D**2 + 1)