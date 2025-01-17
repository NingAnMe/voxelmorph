import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, weights=True):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        if weights:
            weight = np.abs(1 / np.sin(np.linspace(0 + 0.006, np.pi - 0.006, y_pred.shape[-1])))
            weight = np.clip(weight, 0, 10)
            weight = torch.from_numpy(weight).to("cuda")
            cc *= weight

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, weights=True):
        if weights:
            weight = np.abs(1 / np.sin(np.linspace(0 + 0.006, np.pi - 0.006, y_pred.shape[-1])))
            weight = np.clip(weight, 0, 10)
            weight = torch.from_numpy(weight).float().to("cuda")
            mse_loss = torch.mean(((y_true - y_pred) ** 2) * weight)
        else:
            mse_loss = torch.mean((y_true - y_pred) ** 2)
        return mse_loss


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, weights=True):
        if len(y_pred) == 5:
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz
            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        elif len(y_pred) == 4:
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            if weights:  # 1 / sin(phi)
                weight = np.abs(1 / np.sin(np.linspace(0 + 0.006, np.pi - 0.006, y_pred.shape[-1])))
                weight = np.clip(weight, 0, 10)
                weight = torch.from_numpy(weight).float().to("cuda")
                dy = weight * dy

                dx = weight[:-1] * dx

            d = torch.mean(dx) + torch.mean(dy)
        else:
            raise KeyError("must be 3D or 2D")
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class KL:
    """
    Kullback–Leibler divergence for probabilistic flows.
    """

    def __init__(self, prior_lambda, flow_vol_shape=None):
        self.prior_lambda = prior_lambda
        self.flow_vol_shape = flow_vol_shape
        self.D = None

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewhere.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims, dtype=np.float)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filt = np.zeros([ndims, ndims] + [3] * ndims)
        for i in range(ndims):
            filt[i, i, ...] = filt_inner

        return filt

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [ndims, *vol_shape]

        # prepare conv kernel
        conv_fn = getattr(torch.nn.functional, 'conv%dd' % ndims)

        # prepare tf filter
        z = torch.ones([1] + sz)
        filt_tf = torch.from_numpy(self._adj_filt(ndims)).float()
        result = conv_fn(z, filt_tf, stride=1, padding='same')
        return result

    def prec_loss(self, y_pred, weight=None):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.shape[1:-1]
        ndims = len(vol_shape)

        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = torch.permute(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            df = df * df
            if weight is not None:
                df = weight * df
            sm += torch.mean(df)

        return 0.5 * sm / ndims

    def loss(self, y_true, y_pred, weights=True):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.shape) - 2
        mean = y_pred[:, 0:ndims, ...]
        log_sigma = y_pred[:, 0:ndims, ...]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data
        if self.flow_vol_shape is None:
            self.flow_vol_shape = y_pred.shape[2:]
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape).to("cuda")

        # sigma terms
        if weights:  # 1 / sin(phi)
            weight = np.abs(1 / np.sin(np.linspace(0 + 0.006, np.pi - 0.006, y_pred.shape[-1])))
            weight = np.clip(weight, 0, 10)
            weight = torch.from_numpy(weight).float().to("cuda")
            sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma
            sigma_term = weight * sigma_term
            sigma_term = torch.mean(sigma_term)

            # precision terms
            # note needs 0.5 twice, one here (inside self.prec_loss), one below
            prec_term = self.prior_lambda * self.prec_loss(mean, weight)
        else:
            sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma
            sigma_term = torch.mean(sigma_term)

            # precision terms
            # note needs 0.5 twice, one here (inside self.prec_loss), one below
            prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        # ndims because we averaged over dimensions as well
        return 0.5 * ndims * (sigma_term + prec_term)


class MutualInformation:
    """
    Mutual Information
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.loss(y_true, y_pred)
