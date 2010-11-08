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
                pylab.plot(ticks, np.log(np.abs(rec)), ticks, np.log(np.abs(frame)))
                pylab.ylim(-10, 10)
                pylab.show()
            yield rec

def learnit():
    pipe = ronwtools.Pipeline(
        ronwtools.AudioSource('../chess.ogg'),
        ronwtools.Mono(),
        ronwtools.STFT(nfft=16384, nhop=4096, winfun=np.hanning),
        CCIPCAProcessor((8193, 200), plot=False),
        ronwtools.ISTFT(nfft=16384, nhop=4096, winfun=np.hanning),
        ronwtools.Framer(524288)
    )
    for segment in pipe:
        segment /= np.max(segment)
        print np.max(segment)
        audiolab.play(segment)

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

generate()
