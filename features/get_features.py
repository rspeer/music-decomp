import numpy as np
import ronwtools
import pylab
pylab.hold(False)

cqt = ronwtools.CQT(44100, nfft=32768, nhop=1024, fmax=7040, fmin=27.5, bpo=24)
pipe = ronwtools.Pipeline(
    ronwtools.AudioSource('../chess.ogg'),
    ronwtools.Mono(),
    cqt
)
window = np.zeros((88, 100))
for frame in pipe:
    window[:, :-1] = window[:, 1:]
    window[:, -1] = np.abs(frame)[::2]
    pylab.imshow(window, aspect='auto', origin='lower', interpolation='nearest')
    pylab.show()

