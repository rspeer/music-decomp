from plca import SIPLCA2, normalize, logger, plt, shift, EPS, plottools
import numpy as np

class Basilica(SIPLCA2):
    def __init__(self, V, rank, win, circular=False, alphaWf=0, alphaWt=0,
                 betaWf=0, betaWt=0, **kwargs):
        SIPLCA2.__init__(self, V, rank, win, circular, **kwargs)
        
        self.VRWf = np.empty((self.F, self.rank))
        self.VRWt = np.empty((self.rank, self.winT))
        self.VRH = np.empty((self.T, self.rank, self.winF))
        self.alphaWf = 1 + alphaWf
        self.alphaWt = 1 + alphaWt
        self.betaWf = betaWf
        self.betaWt = betaWt

    @staticmethod
    def reconstruct(Wf, Wt, Z, H, norm=1.0, circular=False):
        W = Basilica.w_product(Wf, Wt)
        return SIPLCA2.reconstruct(W, Z, H, norm, circular)

    @staticmethod
    def w_product(Wf, Wt):
        return Wf[:,:,np.newaxis] * Wt[np.newaxis,:,:]

    def initialize(self):
        W, Z, H = SIPLCA2.initialize(self)
        Wf = normalize(np.random.rand(self.F, self.rank), 0)
        Wt = normalize(np.random.rand(self.rank, self.winT), 1)
        return Wf, Wt, Z, H
    
    @classmethod
    def analyze(cls, V, rank, niter=100, convergence_thresh=1e-9,
                printiter=50, plotiter=None, plotfilename=None,
                initWf=None, initWt=None, initZ=None, initH=None,
                updateW=True, updateZ=True, updateH=True, **kwargs):
        norm = V.sum()
        V /= norm
    
        params = cls(V, rank, **kwargs)
        iWf, iWt, iZ, iH = params.initialize()
    
        Wf = iWf if initWf is None else initWf.copy()
        Wt = iWt if initWt is None else initWt.copy()
        Z = iZ if initZ is None else initZ.copy()
        H = iH if initH is None else initH.copy()
        
        params.Wf = Wf
        params.Wt = Wt
        params.Z = Z
        params.H = H
    
        oldlogprob = -np.inf
        for n in xrange(niter):
            logprob, WZH = params.do_estep(Wf, Wt, Z, H)
            if n % printiter == 0:
                logger.info('Iteration %d: logprob = %f', n, logprob)
            if plotiter and n % plotiter == 0:
                params.plot(V, Wf, Wt, Z, H, n)
                if not plotfilename is None:
                    plt.savefig('%s_%04d.png' % (plotfilename, n))
            if logprob < oldlogprob:
                logger.debug('Warning: logprob decreased from %f to %f at '
                             'iteration %d!', oldlogprob, logprob, n)
                #import pdb; pdb.set_trace()
            elif n > 0 and logprob - oldlogprob < convergence_thresh:
                logger.info('Converged at iteration %d', n)
                break
            oldlogprob = logprob
    
            nWf, nWt, nZ, nH = params.do_mstep(n)
    
            if updateW:
                Wf = nWf
                Wt = nWt
            if updateZ:  Z = nZ
            if updateH:  H = nH
    
            params.Wf = Wf
            params.Wt = Wt
            params.Z = Z
            params.H = H

        if plotiter:
            params.plot(V, Wf, Wt, Z, H, n)
            if not plotfilename is None:
                plt.savefig('%s_%04d.png' % (plotfilename, n))
        logger.info('Iteration %d: final logprob = %f', n, logprob)
        recon = norm * WZH
        return Wf, Wt, Z, H, norm, recon, logprob

 
    # W.shape = (F, rank, winT)
    # VRW.shape = (F, rank, winT)
    # H.shape = (rank, winF, T)
    # Hshifted.shape = (rank, T)
    # VRH.shape = (T, rank, winF)
    # WZH.shape = (F, T)
    #
    # Wf.shape = (F, rank)
    # Wt.shape = (rank, winT)
    # VRWf.shape = (F, rank)
    # VRWt.shape = (rank, winT)

    def do_estep(self, Wf, Wt, Z, H):
        WZH = self.reconstruct(Wf, Wt, Z, H, circular=self.circular)
        logprob = self.compute_logprob(Wf, Wt, Z, H, WZH)

        WfZ = Wf * Z[np.newaxis,:]
        WtZ = Wt * Z[:,np.newaxis]

        VdivWZH = (self.V / (WZH + EPS))[:,:,np.newaxis]
        self.VRWf[:] = 0
        self.VRWt[:] = 0
        self.VRH[:] = 0
        for r in xrange(self.winF):
            WfZshifted = shift(WfZ, r, 0, self.circularF)
            for tau in xrange(self.winT):
                Hshifted = shift(H[:,r,:], tau, 1, self.circularT)
                ## Hshifted : (rank, T)
                ## tmp : (F, T, rank)
                ## Previously:
                # tmp = ((WZshifted[:,:,tau][:,:,np.newaxis]
                #         * Hshifted[np.newaxis,:,:]).transpose((0,2,1))
                #        * VdivWZH)
                WZtau = WfZshifted * WtZ[:,tau][np.newaxis,:] # (F, rank)
                tmp = Basilica.w_product(WZtau, Hshifted).transpose((0,2,1)) * VdivWZH
                self.VRWf += shift(tmp.sum(1), -r, 0, self.circularF)
                self.VRWt[:,tau] += tmp.sum(1).sum(0)
                self.VRH[:,:,r] += shift(tmp.sum(0), -tau, 0, self.circularT)

        return logprob, WZH

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wf_evidence = self._fix_negative_values(self.VRWf + self.alphaWf - 1)
        Wt_evidence = self._fix_negative_values(self.VRWt + self.alphaWt - 1)

        initialWf = normalize(Wf_evidence, axis=0)
        Wf = self._apply_entropic_prior_and_normalize(
             initialWf, Wf_evidence, self.betaWf, nu=self.nu, axis=0)

        initialWt = normalize(Wt_evidence, axis=1)
        Wt = self._apply_entropic_prior_and_normalize(
             initialWt, Wt_evidence, self.betaWt, nu=self.nu, axis=1)

        Hevidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
                                              + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=[1, 2])
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=[1, 2])

        return Wf, Wt, Z, H
        #return self._prune_undeeded_bases(Wf, Wt, Z, H, curriter)
    
    def compute_logprob(self, Wf, Wt, Z, H, recon):
        logprob = np.sum(self.V * np.log(recon + EPS*recon))
        # Add Dirichlet and Entropic priors.
        logprob += (np.sum((self.alphaWf - 1) * np.log(Wf + EPS*Wf))
                    + np.sum((self.alphaWt - 1) * np.log(Wt + EPS*Wt))
                    + np.sum((self.alphaZ - 1) * np.log(Z + EPS*Z))
                    + np.sum((self.alphaH - 1) * np.log(H + EPS*H)))
        # Add Entropic priors.
        logprob += (self.betaWf * np.sum(Wf * np.log(Wf + EPS*Wf))
                    + self.betaWt * np.sum(Wt * np.log(Wt + EPS*Wt))
                    + self.betaZ * np.sum(Z * np.log(Z + EPS*Z))
                    + self.betaH * np.sum(H * np.log(H + EPS*H)))
        return logprob

    def plot(self, V, Wf, Wt, Z, H, curriter=-1):
        rank = len(Z)
        nrows = rank + 2
        WZH = self.reconstruct(Wf, Wt, Z, H, circular=self.circular)
        W = Basilica.w_product(Wf, Wt)
        plottools.plotall([V, WZH] + [SIPLCA2.reconstruct(W[:,z,:], Z[z], H[z,:],
                                                       circular=self.circular)
                                      for z in xrange(len(Z))], 
                          title=['V (Iteration %d)' % curriter,
                                 'Reconstruction'] +
                          ['Basis %d reconstruction' % x
                           for x in xrange(len(Z))],
                          colorbar=False, grid=False, cmap=plt.cm.hot,
                          subplot=(nrows, 2), order='c', align='xy')
        plottools.plotall([None] + [Z], subplot=(nrows, 2), clf=False,
                          plotfun=lambda x: plt.bar(np.arange(len(x)) - 0.4, x),
                          xticks=[[], range(rank)], grid=False,
                          colorbar=False, title='Z')

        plots = [None] * (3*nrows + 2)
        titles = plots + ['W%d' % x for x in range(rank)]
        wxticks = [[]] * (3*nrows + rank + 1) + [range(0, W.shape[2], 10)]
        plots.extend(W.transpose((1, 0, 2)))
        plottools.plotall(plots, subplot=(nrows, 6), clf=False, order='c',
                          align='xy', cmap=plt.cm.hot, colorbar=False, 
                          ylabel=r'$\parallel$', grid=False,
                          title=titles, yticks=[[]], xticks=wxticks)
        
        plots = [None] * (2*nrows + 2)
        titles=plots + ['H%d' % x for x in range(rank)]
        if np.squeeze(H).ndim < 4:
            plotH = np.squeeze(H)
        else:
            plotH = H.sum(2)
        if rank == 1:
            plotH = [plotH]
        plots.extend(plotH)
        plottools.plotall(plots, subplot=(nrows, 3), order='c', align='xy',
                          grid=False, clf=False, title=titles, yticks=[[]],
                          colorbar=False, cmap=plt.cm.hot, ylabel=r'$*$',
                          xticks=[[]]*(3*nrows-1) + [range(0, V.shape[1], 100)])
        plt.draw()


    def _prune_undeeded_bases(self, Wf, Wt, Z, H, curriter):
        """Discards bases which do not contribute to the decomposition"""
        threshold = 10 * EPS
        zidx = np.argwhere(Z > threshold).flatten()
        if len(zidx) < self.rank and curriter >= self.minpruneiter:
            logger.info('Rank decreased from %d to %d during iteration %d',
                        self.rank, len(zidx), curriter)
            self.rank = len(zidx)
            Z = Z[zidx]
            Wf = Wf[:,zidx]
            Wt = Wt[zidx,:]
            H = H[zidx,:]
            self.VRWf = self.VRWf[:,zidx]
            self.VRWt = self.VRWt[zidx,:]
            self.VRH = self.VRH[:,zidx]
        return Wf, Wt, Z, H


