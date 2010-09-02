from plca.plca import SIPLCA2, EPS, shift
from plca.basilica import Basilica
from musicproc.analyze import *
from musicproc.harmonic_prior import harmonic_prior
from csc import divisi2
import numpy as np
from matplotlib import pyplot as plt
np.seterr(invalid='raise')

#pitch = divisi2.load('clocks.pitch.pickle')
analyzer = MusicAnalyzer(window_size=44100, subsample=1470)
audio = AudioData.from_file('high-hopes.ogg')
pitch = analyzer.quantize_equal(np.abs(analyzer.analyze_pitch(audio, 120)), 1470)
divisi2.save(pitch, 'high-hopes.pitch.pickle')

alphaWf = harmonic_prior(96, 0, 6, 0)
alphaWt = np.vstack([np.linspace(0.01, 0, 30)]*6)

bas = Basilica(pitch, 6, 30, alphaWf=alphaWf, alphaWt=alphaWt, betaHf=0.1, betaHt=0.05)

def play_reconstruction(rec):
    analyzer.reconstruct_W(rec).play()

Wf, Zf, Hf, Wt, Zt, Ht, meta_Hf, meta_Ht, rec = bas.run(pitch, nsubiter=20, niter=1, play_func=play_reconstruction)

