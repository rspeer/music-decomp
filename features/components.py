from csc.divisi2.fast_ccipca import CCIPCA
from csc import divisi2
import ronwtools
from ronwtools import dataprocessor
import numpy as np
from scikits import audiolab

import pylab
#pylab.hold(False)
ticks = np.log(np.arange(1, 8194))

class CCIPCAProcessor(dataprocessor.DataProcessor):
    def __init__(self, shape, plot=False, **kwargs):
        self.plot = plot
        self.ccipca = CCIPCA(np.zeros(shape, np.complex128), **kwargs)
        self.output_ccipca = self.ccipca

    def process_sequence(self, frames):
        for frame in frames:
            mags = self.ccipca.learn_vector(frame)
            mags *= np.abs(mags)
            rec = self.output_ccipca.reconstruct(mags)
            rec[0] = 0
            rec /= (np.linalg.norm(rec) / np.linalg.norm(frame))
            
            if self.plot:
                #pylab.plot(ticks, np.log(np.abs(rec)), ticks, np.log(np.abs(frame)))
                pylab.bar(xrange(10), np.abs(mags[:10]))
                pylab.ylim(0, 1000)
                pylab.show()
            yield rec

def learnit():
    ccipca = CCIPCAProcessor((8193, 400), plot=False, amnesia=1.0)
    pipe = ronwtools.Pipeline(
        ronwtools.AudioSource('../chess.ogg'),
        ronwtools.Mono(),
        ronwtools.STFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ccipca,
        ronwtools.ISTFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ronwtools.Framer(524288)
    )
    for segment in pipe:
        segment /= np.max(segment)
        print np.max(segment)
        audiolab.play(segment)
    divisi2.save(ccipca.ccipca.matrix, 'chess2.eigs')

def generate():
    ccipca = CCIPCAProcessor((8193, 200), plot=False)
    ccipca.output_ccipca = CCIPCA(divisi2.load('chess.eigs'))
    pipe = ronwtools.Pipeline(
        ronwtools.AudioSource('../koyaanisqatsi.ogg'),
        ronwtools.Mono(),
        ronwtools.STFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ccipca,
        ronwtools.ISTFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ronwtools.Framer(1048576)
    )
    for segment in pipe:
        print np.max(segment)
        segment /= np.max(segment)
        audiolab.play(segment)

def correlate():
    ccipca = CCIPCAProcessor((8193, 200), plot=False, amnesia=1.0)
    ccipca.ccipca = CCIPCA(divisi2.load('chess.eigs'))
    ccipca.ccipca.iteration = 100000
    ccipca.output_ccipca = ccipca.ccipca
    pipe = ronwtools.Pipeline(
        ronwtools.AudioSource('../chess.ogg'),
        ronwtools.Mono(),
        ronwtools.STFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ccipca,
        ronwtools.ISTFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ronwtools.Framer(1048576)
    )
    for segment in pipe:
        print np.max(segment)
        segment /= np.max(segment)
        audiolab.play(segment)


correlate()
