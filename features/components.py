from stft import make_pipe, stream_plot
from csc.divisi2.fast_ccipca import CCIPCA
import ronwtools
from ronwtools import dataprocessor
import numpy as np
from scikits import audiolab

import pylab
pylab.hold(False)
ticks = np.log(np.arange(1, 8194))

class CCIPCAProcessor(dataprocessor.DataProcessor):
    def __init__(self, shape, plot=False, **kwargs):
        self.plot = plot
        self.ccipca = CCIPCA(np.zeros(shape, np.complex128), **kwargs)

    def process_sequence(self, frames):
        for frame in frames:
            mags = self.ccipca.learn_vector(frame)
            mags[0] = 0
            rec = self.ccipca.reconstruct(mags)
            rec /= (np.linalg.norm(rec) / np.linalg.norm(frame))
            
            if self.plot:
                pylab.plot(ticks, np.log(np.abs(rec)), ticks, np.log(np.abs(frame)))
                pylab.ylim(-10, 10)
                pylab.show()
            yield rec

pipe = ronwtools.Pipeline(
    make_pipe('../high-hopes.ogg'),
    CCIPCAProcessor((8193, 100), plot=False),
    ronwtools.ISTFT(nfft=16384, nhop=8192, winfun=np.ones),
    ronwtools.Framer(262144)
)
for segment in pipe:
    segment /= np.max(segment)
    print np.max(segment)
    audiolab.play(segment)

