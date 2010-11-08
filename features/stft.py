import numpy as np
import ronwtools

ticks = np.log(np.arange(1, 8194))

def make_pipe(filename='../high-hopes.ogg', nfft=16384, nhop=16384, winfun=np.hanning):
    stft = ronwtools.STFT(nfft=nfft, nhop=nhop, winfun=winfun)
    pipe = ronwtools.Pipeline(
        ronwtools.AudioSource(filename),
        ronwtools.Mono(),
        stft
    )
    return pipe

def stream_plot(pipe):
    import pylab
    pylab.hold(False)
    for frame in pipe:
        pylab.plot(ticks, np.log(np.abs(frame)))
        pylab.ylim(-10, 10)
        pylab.show()

