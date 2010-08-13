from plca import *
import numpy as np

HARMONICS = [0, 12, 19, 24, 28, 31, 34, 36, 38, 40, 43, 46, 47, 48]
NHARMONICS = len(HARMONICS)
HARMONIC_VALUES = [(1, 0),
                   (2, 12),
                   (3, 19),
                   (4, 24),
                   (5, 28),
                   (6, 31),
                   (8, 36),
                   (9, 38),
                   (10, 40),
                   (12, 43),
                   (15, 47),
                   (16, 48),
                   (18, 50),
                   (20, 52),
                   (24, 55),
                   (27, 57),
                   (30, 59),
                   (32, 60)]
HARMONIC_VEC = np.zeros((96,)) + EPS
HARMONIC_VEC2 = np.zeros((96,)) + EPS
HARMONIC_VEC3 = np.zeros((96,)) + EPS
for harmonic, steps in HARMONIC_VALUES:
    HARMONIC_VEC[steps] = 1.0
    HARMONIC_VEC2[steps] = 1.0/harmonic
    if harmonic % 2 == 1:
        HARMONIC_VEC3[steps] = 1.0/((harmonic+1)/2)
HARMONIC_VEC /= np.sum(HARMONIC_VEC)
HARMONIC_VEC2 /= np.sum(HARMONIC_VEC2)
HARMONIC_VEC3 /= np.sum(HARMONIC_VEC3)
NON_HARMONIC_VEC = 1.0/np.exp(np.arange(1, 97))
NON_HARMONIC_VEC /= np.sum(NON_HARMONIC_VEC)

def fixed_shiplca(matrix, **kwargs):
    W = np.zeros((96, 3, 1))
    W[:, 0, :] = NON_HARMONIC_VEC[:, np.newaxis]
    W[:, 1, :] = HARMONIC_VEC2[:, np.newaxis]
    W[:, 2, :] = HARMONIC_VEC3[:, np.newaxis]
    Z = np.ones((3,))/3.0
    results = SHIPLCA.analyze(matrix, initW=W, updateW=False,
                              initZ=Z, updateZ=False, **kwargs)
    return results

class SHIPLCA(SIPLCA2):
    """
    It stands for Shift/Harmonic Invariant Probabilistic Latent Component
    Analysis. It's a 2-D SIPLCA that knows about the harmonic series.
    """
    def initialize(self):
        W, Z, H = super(SIPLCA2, self).initialize()
        W = np.random.rand(self.F, self.rank, self.winT)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]

        # fix to harmonics
        #W *= HARMONIC_VEC[:, np.newaxis, np.newaxis]
        #W[0] = 0.1
        #W[1:] /= np.sum(W[1:]/0.1, axis=0)[np.newaxis,:,:]
        
        H = np.random.rand(self.rank, self.winF, self.T)
        H /= H.sum(2).sum(1)[:,np.newaxis,np.newaxis]
        return W, Z, H

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wevidence = self._fix_negative_values(self.VRW + self.alphaW - 1)[HARMONICS,:,:]
        initialW = normalize(Wevidence, axis=[0, 2])
        smallW = self._apply_entropic_prior_and_normalize(
                 initialW, Wevidence, self.betaW, nu=self.nu, axis=[0, 2])

        smallW[0] = 0.1
        smallW[1:] /= np.sum(smallW[1:]/0.1, axis=0)[np.newaxis,:,:]
        W = np.zeros(self.VRW.shape) + EPS
        W[HARMONICS,:,:] = smallW
        
        Hevidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
                                              + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=[1, 2])
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=[1, 2])

        return self._prune_undeeded_bases(W, Z, H, curriter)
