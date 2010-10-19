# Search for a deconvolution filter D
# Minimize ||D * V||^2
import numpy as np
from numpy import newaxis
import pylab
from scipy import signal
from scipy.signal.signaltools import _centered
EPS = 1e-6

def sparsify(mat, s):
    return mat * (np.argsort(mat, axis=1) < s)

def normalize(A, axis=None):
    if axis is None:
        return A / np.sum(np.abs(A))
    if isinstance(axis, int):
        axis = (axis,)
        
    norm = np.abs(A)
    for ax in reversed(sorted(axis)):
        norm = norm.sum(ax)
    nshape = np.array(A.shape)
    nshape[axis] = 1
    norm.shape = nshape
    return A / (norm + EPS)


def magical_entropy_hammer(param, evidence, beta, nu=50, niter=30, convergence_thresh=1e-7, axis=None):
    """
    Do something like PLCA's _apply_entropic_prior_and_normalize to encourage
    sparse data. However, make it work on negative numbers as well, and
    normalize by the Euclidean norm.

    I have no idea what this actually means, mathematically.

    The name of the function is meant to convey the idea of "just hitting your
    data with the magical entropy hammer until it's sparse".
    """
    for i in xrange(niter):
        lastparam = param.copy()
        magnified = (np.abs(param) ** (nu / (nu-1.0))) * np.sign(param)
        alpha = normalize(magnified, axis)
        param = normalize(evidence + beta * nu * alpha, axis)
        if np.mean(np.abs(param - lastparam)) < convergence_thresh:
            break
    return param

def fft_convolve(v1, v2):
    s1 = v1.shape[-1]
    s2 = v2.shape[-1]
    valid = abs(s2-s1) + 1
    fsize = 2**np.ceil(np.log2(s1+s2-1))

    convolver = (EPS + signal.fft(v2, fsize, axis=-1))
    convolved = np.real(signal.ifft(convolver * signal.fft(v1, fsize, axis=-1), axis=-1))
    return _centered(convolved, valid)

def fft_deconvolve(num, denom):
    s1 = num.shape[-1]
    s2 = denom.shape[-1]
    valid = abs(s2-s1) + 1
    fsize = 2**np.ceil(np.log2(s1+s2-1))

    deconvolver = 1.0/(EPS + signal.fft(denom, fsize, axis=-1))
    deconvolved = np.real(signal.ifft(deconvolver * signal.fft(num, fsize, axis=-1), axis=-1))
    return deconvolved[..., :valid]

def degrades(V, W, H, rate, s, niter=20):
    """
    Deconvolution by Gradient Descent with Sparsification.
    """
    V, W, H = normalize(V), normalize(W), normalize(H)
    for iter in xrange(niter):
        convolved_pieces = np.vstack([signal.convolve(W[r], H[r], mode='full') for r in xrange(W.shape[0])])
        convolved = np.sum(convolved_pieces, axis=0)
        delta = V - convolved
        projected_H = fft_deconvolve(delta, W)
        H = magical_entropy_hammer(H, H + projected_H * rate, 0.01, axis=1)
        
        
        convolved_pieces = np.vstack([signal.convolve(W[r], H[r], mode='full') for r in xrange(W.shape[0])])
        convolved = np.sum(convolved_pieces, axis=0)
        delta = V - convolved
        projected_W = fft_deconvolve(delta, H)
        W = magical_entropy_hammer(W, W + projected_W * rate, 0.001, axis=1)
        
        print np.sum(np.abs(delta))
        print H
        
    pylab.clf()
    pylab.subplot(311)
    pylab.plot(W.T)
    pylab.subplot(312)
    pylab.plot(H.T)
    pylab.subplot(313)
    pylab.plot(V)
    pylab.plot(convolved)
    return W, H

if __name__ == '__main__':
    func = np.concatenate([
        np.linspace(0.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, 1.0, 10, endpoint=False),
        np.linspace(1.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, -1.0, 10, endpoint=False),
        np.linspace(-1.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, 0.0, 10, endpoint=False),
        np.linspace(0.0, 1.0, 10, endpoint=False),
        np.linspace(1.0, -1.0, 10, endpoint=False),
        np.linspace(-1.0, 0, 10, endpoint=False),
        np.linspace(0.0, 0.0, 10, endpoint=False),
    ], axis=1)

    initH = np.random.normal(size=(1, 100))
    initW = np.random.normal(size=(1, 21))
    W, H = degrades(func, initW, initH, 0.1, 5, 500)

