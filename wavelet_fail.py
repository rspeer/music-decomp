from scipy.signal import fftconvolve, wavelets, resample, hamming
from scipy.fftpack import fft, ifft
import numpy as np
from scikits import audiolab
RATE = 44100

C0 = 16.3516
M = 65536 * 4
nfilters = 96
def morlet_freq(f0, M):
    """
    Get a Morlet wavelet tuned to detect a particular frequency.
    """
    # this version scales the length of the waveform
    # return wavelets.morlet(RATE*40/f0, 40, 0.5)

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    return w / np.linalg.norm(w)

def downsample(signal, factor):
    return resample(signal, signal.shape[-1]/factor)

def modified_hamming(n):
    h = hamming(n)
    modifier = h[:n/2] + h[n/2:]
    return h / np.concatenate([modifier, modifier])

filters = np.zeros((nfilters, M), dtype='complex128')
fftfilters = np.zeros((nfilters, M), dtype='complex128')
for x in xrange(nfilters):
    filters[x] = morlet_freq(C0 * 2.0**(x/12.0), M)
    fftfilters[x] = fft(filters[x])
triangle = modified_hamming(M)
print "made filters"

# global so we don't have to reallocate it over and over
fftsignal = np.zeros((M,), dtype='complex128')

def wavelet_detect(signal):
    global fftsignal
    #fftsignal[:] = fft(signal * triangle)
    fftsignal[:] = fft(signal)
    convolved = ifft(fftfilters * fftsignal)
    return np.abs(convolved)

def stream_wavelets(signal):
    sig = np.concatenate([np.zeros((M/2,)),
                          signal,
                          np.zeros((M,))])
    lastchunk = np.zeros((nfilters, M/2))
    for frame in xrange(sig.shape[-1]/M):
        chunk = wavelet_detect(sig[frame*M: (frame+1)*M])
        yield chunk.copy()
    #for frame in xrange(sig.shape[-1]*2/M - 1):
    #    chunk = wavelet_detect(sig[frame*M/2 : (frame+2)*M/2]) * triangle
    #    lastchunk[:] = chunk[:, M/2:]
    #    yield lastchunk.copy() + chunk[:, :M/2]

if __name__ == '__main__':
    import pylab, time
    sndfile = audiolab.Sndfile('clocks.ogg')
    signal = np.mean(sndfile.read_frames(44100*20), axis=1)
    pylab.figure(1)
    graph_pieces = []
    for output in stream_wavelets(signal):
        print time.time()
        graph_pieces.append(output)
    wavelet_graph = np.concatenate(graph_pieces[1:-1], axis=-1)
    pylab.imshow(wavelet_graph[:, ::10], aspect='auto', origin='lower')
    pylab.show()

#mfcc = np.abs(fft(np.log(output[16:112].T)))
#pylab.figure(2)
#pylab.imshow(mfcc[::10, 1:65].T, aspect='auto', origin='lower')

