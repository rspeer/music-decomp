from plca.plca import SIPLCA2, EPS, shift
from plca.basilica import Basilica
from musicproc.analyze import *
from musicproc.harmonic_prior import harmonic_prior
from csc import divisi2
import numpy as np
from matplotlib import pyplot as plt
np.seterr(invalid='raise')

pitch = divisi2.load('chess.pitch.pickle')
analyzer = MusicAnalyzer(window_size=44100, subsample=735)
#audio = AudioData.from_file('chess.ogg')
#pitch = analyzer.quantize_equal(np.abs(analyzer.analyze_pitch(audio, 20)), 735)
#divisi2.save(pitch, 'chess.pitch.pickle')

alphaWf = harmonic_prior(96, 0, 4, 0)
alphaWt = np.vstack([np.linspace(0.01, 0, 60)]*4)

bas = Basilica(pitch, 4, 60, alphaWf=alphaWf, alphaWt=alphaWt, betaHf=0.05, betaHt=0.05)

def play_reconstruction(rec):
    analyzer.reconstruct_W(rec).play()

Wf, Zf, Hf, Wt, Zt, Ht, meta_Wf, meta_Zf, meta_Hf = bas.run(pitch, nsubiter=8, niter=8, play_func=play_reconstruction)

