from plca import *
import numpy as np

HARMONICS = [0, 12, 19, 24, 28, 31, 34, 36, 38, 40]
class SHIPLCA(SIPLCA2):
    """
    It stands for Shift/Harmonic Invariant Probabilistic Latent Component
    Analysis. It's a 2-D SIPLCA that knows about the harmonic series.
    """
    def initialize(self):
        W, Z, H = super(SIPLCA2, self).initialize()
        W = np.random.rand(self.F, self.rank, self.winT)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]

        # Rob: fixed baseline
        W[0] = 0.1
        W[1:] /= np.sum(W[1:]/0.1, axis=0)[np.newaxis,:,:]
        
        H = np.random.rand(self.rank, self.winF, self.T)
        H /= H.sum(2).sum(1)[:,np.newaxis,np.newaxis]
        return W, Z, H

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wevidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialW = normalize(Wevidence, axis=[0, 2])
        W = self._apply_entropic_prior_and_normalize(
            initialW, Wevidence, self.betaW, nu=self.nu, axis=[0, 2])

        # Rob: fixed baseline
        W[0] = 0.1
        W[1:] /= np.sum(W[1:]/0.1, axis=0)[np.newaxis,:,:]
        
        Hevidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
                                              + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=[1, 2])
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=[1, 2])

        return self._prune_undeeded_bases(W, Z, H, curriter)
