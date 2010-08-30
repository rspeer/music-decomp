from plca.plca import SIPLCA2, EPS, shift
from plca.basilica import Basilica
from musicproc.analyze import *
from musicproc.harmonic_prior import harmonic_prior
from csc import divisi2
import numpy as np
from matplotlib import pyplot as plt
np.seterr(invalid='raise')

pitch = divisi2.load('clocks.pitch.pickle')
analyzer = MusicAnalyzer(window_size=44100, subsample=1470)
#audio = AudioData.from_file('clocks.ogg')
#pitch = analyzer.quantize_equal(np.abs(analyzer.analyze_pitch(audio, 20)), 1470)
#divisi2.save(pitch, 'clocks.pitch.pickle')

alphaWf = harmonic_prior(96, 0, 4, 0)
alphaWt = np.vstack([np.linspace(0.01, 0, 30)]*4)

bas = Basilica(pitch, 4, 30, alphaWf=alphaWf, alphaWt=alphaWt, betaH=0.1)

plt.plot(alphaWf[...,0])
plt.show()
Wf, Zf, Hf, Wt, Zt, Ht, meta_Wf, meta_Zf, meta_Hf = bas.run(pitch)

