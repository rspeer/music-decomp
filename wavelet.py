from scipy.signal import fftconvolve, wavelets, resample, hanning, chirp
from scipy.fftpack import fft, ifft
import numpy as np
from scikits import audiolab
from csc import divisi2
RATE = 44100

A0 = 27.5
M = 65536
nfilters = 96
def morlet_freq(f0, M):
    """
    Get a Morlet wavelet tuned to detect a particular frequency.
    """
    # this version scales the length of the waveform
    # return wavelets.morlet(RATE*40/f0, 40, 0.5)

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    return w * (f0/RATE) / np.linalg.norm(w)

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
    filter1 = morlet_freq(A0 * 2.0**(x/12.0), M)
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
        yield (lastchunk + chunk[:, :M/2])
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
    U, V = d.nmf(k=40)
    U = np.asarray(U)
    V = np.asarray(V)
    U_prime = np.zeros(U.shape)
    for col in xrange(U.shape[1]):
        row = np.argmax(U[:,col])
        U_prime[row, col] = 1 #U[row,col]
    return np.dot(U**3, V.T)

ALPHA = 0.5
meansq = 0.00001
def timbre_color(matrix):
    # okay, this really is the evil kind of global var. I'll classify sometime
    global meansq
    matrix = np.concatenate([np.zeros((43, matrix.shape[-1])), matrix])
    deharmonized = matrix[43:]
    for h, steps in enumerate((12, 19, 24, 28, 31, 34, 36, 38, 40, 42, 43)):
        harmonic = h + 2
        deharmonized -= matrix[43-steps:-steps]/harmonic
    bigger = np.concatenate([np.zeros((24, deharmonized.shape[-1])), deharmonized])
    color = np.zeros(deharmonized.shape + (3,))
    baseline = np.maximum(0, deharmonized)
    color[:,:,0] = (deharmonized + np.minimum(deharmonized/2, bigger[12:nfilters+12]))
    color[:,:,1] = (deharmonized + np.minimum(deharmonized/3, bigger[19:nfilters+19]))
    color[:,:,2] = (deharmonized + np.minimum(deharmonized/4, bigger[24:nfilters+24]))
    color -= np.mean(color, axis=-1)[...,np.newaxis]
    color = (color*100)
    color = color + 0.75
    rgb = np.maximum(0, baseline[...,np.newaxis] * color)
    prev_meansq = meansq
    meansq = (meansq*(1.0-ALPHA)) + (np.mean(rgb*rgb) * ALPHA)
    meansq_smooth = np.linspace(np.sqrt(prev_meansq), np.sqrt(meansq), matrix.shape[-1])
    return np.minimum(rgb/meansq_smooth[np.newaxis, :, np.newaxis]/10, 1)

def smooth(matrix, n=20):
    end = matrix.shape[1]-n
    result = matrix[:, 0:end]
    for i in xrange(n):
        result = np.maximum(result, matrix[:, i:end+i])
    return result

if __name__ == '__main__':
    import pylab, time
    sndfile = audiolab.Sndfile('clocks.ogg')
    signal = np.mean(sndfile.read_frames(44100*60), axis=1)
    #signal = chirp(np.arange(2**18), 16.3516/44100, 2**18, 4185.01/44100,
    #               method='logarithmic')
    pieces = []
    for piece in stream_wavelets(signal):
        print time.time()
        pieces.append(timbre_color(np.abs(piece[:, ::200])))

    wavelet_graph = np.concatenate(pieces[1:-2], axis=1)
    svdgraph = wavelet_graph
    pylab.figure(1)
    pylab.imshow(svdgraph, aspect='auto', origin='lower')
    pylab.show()

#mfcc = np.abs(fft(np.log(output[16:112].T)))
#pylab.figure(2)
#pylab.imshow(mfcc[::10, 1:65].T, aspect='auto', origin='lower')

