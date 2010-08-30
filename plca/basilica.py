from plca import SIPLCA, SIPLCA2, normalize, logger, plt, shift, EPS, plottools
import numpy as np
from numpy import newaxis
class Basilica(object):
    def __init__(self, V, rank, t_win, alphaWf=0, alphaWt=0, betaWf=0, betaWt=0,
                 betaH=0):
        # Find out if other H parameters are necessary?
        # TODO: incremental V
        self.V = V / V.sum()
        self.f_steps, self.t_steps = self.V.shape
        self.t_win = t_win
        self.rank = rank
        H = np.concatenate([self.V]*rank, axis=1)
        Hflat = H.flatten()[newaxis,:]
        Vflat = self.V.flatten()[newaxis,:]

        # Make the time-independent frequency analyzer
        self.freq_analyzer = SIPLCA2(H, rank, (self.f_steps, 1),
                                     alphaW=alphaWf[newaxis,...],
                                     betaW=betaWf, betaH=betaH)
        self.meta_freq_analyzer = SIPLCA2(self.V, rank, (self.f_steps, 1),
                                          alphaW=alphaWf[newaxis,...],
                                          betaW=betaWf, betaH=betaH)

        # Make the frequency-independent time-envelope analyzer
        self.time_analyzer = SIPLCA(Hflat, rank, self.t_win,
                                    alphaW=alphaWt[newaxis,...],
                                    betaW=betaWt, betaH=betaH)
        self.meta_time_analyzer = SIPLCA(Vflat, rank, self.t_win,
                                         alphaW=alphaWt[newaxis,...],
                                         betaW=betaWt, betaH=betaH)

    def run_freq(self, Vf, Wf=None, Zf=None, Hf=None, niter=5):
        self.freq_analyzer.V = Vf
        if Wf is None or Zf is None or Hf is None:
            initW, initZ, initH = self.freq_analyzer.initialize()
            if Wf is None: Wf = initW
            if Zf is None: Zf = initZ
            if Hf is None: Hf = initH
        
        for iter in xrange(niter):
            logprob, WZH = self.freq_analyzer.do_estep(Wf, Zf, Hf)
            logger.info('Iteration f%d: logprob = %f', iter, logprob)
            Wf, Zf, Hf = self.freq_analyzer.do_mstep(iter)
        self.freq_analyzer.plot(Vf, Wf, Zf, Hf, iter)
        
        meta_logprob, meta_WZH = self.meta_freq_analyzer.do_estep(Wf, Zf, Hf)
        meta_Wf, meta_Zf, meta_Hf = self.meta_freq_analyzer.do_mstep(0)
        return Wf, Zf, Hf, meta_Wf, meta_Zf, meta_Hf
    
    # now do the same for run_time
    def run_time(self, Vt, Wt=None, Zt=None, Ht=None, niter=5):
        self.time_analyzer.V = Vt
        if Wt is None or Zt is None or Ht is None:
            initW, initZ, initH = self.time_analyzer.initialize()
            if Wt is None: Wt = initW
            if Zt is None: Zt = initZ
            if Ht is None: Ht = initH

        for iter in xrange(niter):
            logprob, WZH = self.time_analyzer.do_estep(Wt, Zt, Ht)
            logger.info('Iteration t%d: logprob = %f', iter, logprob)
            Wt, Zt, Ht = self.time_analyzer.do_mstep(iter)
            assert Wt.ndim == 3
            assert Zt.ndim == 1
            assert Ht.ndim == 2
        self.time_analyzer.plot(Vt, Wt, Zt, Ht, iter)

        meta_logprob, meta_WZH = self.meta_time_analyzer.do_estep(Wt, Zt, Ht)
        meta_Wt, meta_Zt, meta_Ht = self.meta_time_analyzer.do_mstep(0)
        return Wt, Zt, Ht, meta_Wt, meta_Zt, meta_Ht

    def run(self, V, niter=10, nsubiter=5, Wf=None, Zf=None, Hf=None, Wt=None, Zt=None, Ht=None):
        meta_Hf = np.dstack([np.concatenate([self.V]*self.rank, axis=1)] * self.rank).transpose(2,0,1)
        for iter in xrange(niter):
            # run_freq returns Hf = (rank, F, rank*T)
            # sum to get (rank, F, T)
            # transpose to get (F, rank, T)
            temp = self.sum_pieces(meta_Hf).transpose(1,0,2)
            # flatten to get (1, F*rank*T)
            Vt = temp.flatten()[newaxis,:]
            
            Wt, Zt, Ht, meta_Wt, meta_Zt, meta_Ht =\
              self.run_time(Vt, Wt, Zt, Ht, nsubiter)

            # run_time returns Ht = (rank, F*rank*T)
            # reshape to get (rank, F, rank*T)
            temp = Ht.reshape(self.rank, self.fsteps, self.rank*self.tsteps)
            # sum to get (rank, F, T)
            # transpose to get (F, rank, T)
            temp = self.sum_pieces(temp).transpose(1,0,2)
            # reshape to get (F, rank*T)
            Vf = np.reshape(temp, (self.fsteps, self.rank*self.tsteps))

            Wf, Zf, Hf, meta_Wf, meta_Zf, meta_Hf =\
              self.run_freq(Vf, Wf, Zf, Hf, nsubiter)
        return Wf, Zf, Hf, Wt, Zt, Ht, meta_Wf, meta_Zf, meta_Hf
    
    def sum_pieces(self, array):
        """
        Given an array made of `r` equal-sized pieces that are concatenated
        along the array's last axis, return the smaller array that results
        from summing these pieces. `r` is defined to be `self.rank`.
        """
        assert array.shape[-1] % self.rank == 0
        width = array.shape[-1] // self.rank
        shape = list(array.shape)
        shape[-1] = width
        result = np.zeros(tuple(shape))
        for i in xrange(self.rank):
            result += array[..., i*width : (i+1)*width]
        return result

class OldBasilica(SIPLCA2):
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
        return Wf[:,:,newaxis] * Wt[newaxis,:,:]

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

        WfZ = Wf * Z[newaxis,:]
        WtZ = Wt * Z[:,newaxis]

        VdivWZH = (self.V / (WZH + EPS))[:,:,newaxis]
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
                # tmp = ((WZshifted[:,:,tau][:,:,newaxis]
                #         * Hshifted[newaxis,:,:]).transpose((0,2,1))
                #        * VdivWZH)
                WZtau = WfZshifted * WtZ[:,tau][newaxis,:] # (F, rank)
                tmp = Basilica.w_product(WZtau, Hshifted).transpose((0,2,1)) * VdivWZH
                self.VRWf += shift(tmp.sum(1), -r, 0, self.circularF)
                self.VRWt[:,tau] += tmp.sum(1).sum(0)
                self.VRH[:,:,r] += shift(tmp.sum(0), -tau, 0, self.circularT)

        return logprob, WZH

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRWf.sum(0)
                                              + self.VRWt.sum(1)
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

        return self._prune_undeeded_bases(Wf, Wt, Z, H, curriter)
    
    def compute_logprob(self, Wf, Wt, Z, H, recon):
        logprob = np.sum(self.V * np.log(recon + EPS))
        # Add Dirichlet and Entropic priors.
        logprob += (np.sum((self.alphaWf - 1) * np.log(Wf + EPS))
                    + np.sum((self.alphaWt - 1) * np.log(Wt + EPS))
                    + np.sum((self.alphaZ - 1) * np.log(Z + EPS))
                    + np.sum((self.alphaH - 1) * np.log(H + EPS)))
        # Add Entropic priors.
        logprob += (self.betaWf * np.sum(Wf * np.log(Wf + EPS))
                    + self.betaWt * np.sum(Wt * np.log(Wt + EPS))
                    + self.betaZ * np.sum(Z * np.log(Z + EPS))
                    + self.betaH * np.sum(H * np.log(H + EPS)))
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
            if isinstance(self.alphaWf, np.ndarray):
                self.alphaWf = self.alphaWf[:,zidx]
            if isinstance(self.alphaWt, np.ndarray):
                self.alphaWt = self.alphaWt[zidx,:]
            if isinstance(self.alphaH, np.ndarray):
                self.alphaH = self.alphaH[zidx,:]
            if isinstance(self.alphaZ, np.ndarray):
                self.alphaZ = self.alphaZ[zidx]
        return Wf, Wt, Z, H


