from scipy.signal import fftconvolve, wavelets, resample, hanning, chirp
from scipy.fftpack import fft, ifft
import numpy as np
from scikits import audiolab
from csc import divisi2
RATE = 44100

C0 = 16.3516
M = 65536
nfilters = 96
def morlet_freq(f0, M):
    """
    Get a Morlet wavelet tuned to detect a particular frequency.
    """
    # this version scales the length of the waveform
    # return wavelets.morlet(RATE*40/f0, 40, 0.5)

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    return w / np.linalg.norm(w)

def morlet_freq_harmonic(f0, M):
    # this doesn't actually work

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    for harmonic in xrange(2, 9):
        w += wavelets.morlet(M, 40, float(f0*M*harmonic)/(RATE*80))/harmonic
        w -= wavelets.morlet(M, 40, float(f0*M)/(harmonic*RATE*80))/harmonic
    return w / np.linalg.norm(w)

def downsample(signal, factor):
    return resample(signal, signal.shape[-1]/factor)

fftfilters = np.zeros((nfilters, M), dtype='complex128')
for x in xrange(nfilters):
    filter1 = morlet_freq(C0 * 2.0**(x/12.0), M)
    fftfilters[x] = fft(filter1)
hanning_window = hanning(M)
print "made filters"

# global so we don't have to reallocate it over and over
fftsignal = np.zeros((M,), dtype='complex128')

def wavelet_detect(signal):
    """
    Uses a set of wavelet filters to detect pitches in the signal, with
    constant pitch resolution (higher than a typical STFT!)
    
    The inevitable tradeoff comes in the form of time resolution in the bass
    notes.
    """
    global fftsignal
    fftsignal[:] = fft(signal * hanning_window).conj()
    convolved = np.roll(ifft(fftfilters * fftsignal), M/2, axis=-1)[:, ::-1]
    return convolved * hanning_window

def stream_wavelets(signal):
    sig = np.concatenate([np.zeros((M/2,)),
                          signal,
                          np.zeros((M,))])
    lastchunk = np.zeros((nfilters, M/2), dtype='complex128')
    for frame in xrange(sig.shape[-1]*2/M - 1):
        chunk = wavelet_detect(sig[frame*M/2 : (frame+2)*M/2])
        yield (lastchunk.copy() + chunk[:, :M/2])
        lastchunk[:] = chunk[:, M/2:]

def windowed_wavelets(signal):
    sig = np.concatenate([np.zeros((M/2,)),
                          signal,
                          np.zeros((M,))])
    output = np.zeros((nfilters, sig.shape[-1]))
    for frame in xrange(sig.shape[-1]*2/M - 1):
        chunk = wavelet_detect(sig[frame*M/2 : (frame+2)*M/2])
        output[:, frame*M/2 : (frame+2)*M/2] += chunk
    return output[:, M/2:-M]

def svd_reduce(matrix):
    d = divisi2.DenseMatrix(matrix)
    U, V = d.nmf(k=20)
    U = np.asarray(U)
    V = np.asarray(V)
    U_prime = np.zeros(U.shape)
    for col in xrange(U.shape[1]):
        row = np.argmax(U[:,col])
        U_prime[row, col] = 1 #U[row,col]
    return np.dot(U_prime, V.T)

def deharmonize_pitches(matrix):
    matrix = np.concatenate([np.zeros((36, matrix.shape[-1])), matrix])
    return np.maximum(0,
      matrix[36:] - matrix[24:-12]/4 - matrix[17,-19]/9 - matrix[12:-24]/3 -
      matrix[8:-28]/16)

if __name__ == '__main__':
    import pylab, time
    sndfile = audiolab.Sndfile('clocks.ogg')
    signal = np.mean(sndfile.read_frames(44100*20), axis=1)
    #signal = chirp(np.arange(2**18), 16.3516/44100, 2**18, 4185.01/44100,
    #               method='logarithmic')
    pieces = []
    for piece in stream_wavelets(signal):
        print time.time()
        pieces.append(piece)

    wavelet_graph = np.abs(np.concatenate(pieces[1:-2], axis=-1))
    svdgraph = deharmonize_pitches(wavelet_graph[:, ::100])
    pylab.figure(1)
    pylab.imshow(svdgraph, aspect='auto', origin='lower')
    pylab.show()

#mfcc = np.abs(fft(np.log(output[16:112].T)))
#pylab.figure(2)
#pylab.imshow(mfcc[::10, 1:65].T, aspect='auto', origin='lower')

