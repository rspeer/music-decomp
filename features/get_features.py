import numpy as np
import ronwtools
import pylab
pylab.hold(False)

cqt = ronwtools.CQT(44100, nfft=32768, nhop=1024, fmax=7040, fmin=27.5, bpo=12)
pipe = ronwtools.Pipeline(
    ronwtools.AudioSource('../high-hopes.ogg'),
    ronwtools.Mono(),
    cqt
)
window = np.zeros((96, 100))
for frame in pipe:
    window[:, :-1] = window[:, 1:]
    window[:, -1] = np.abs(frame)
    pylab.imshow(window, aspect='auto', origin='lower', interpolation='nearest')
    pylab.show()

