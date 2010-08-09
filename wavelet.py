from scipy.signal import fftconvolve, wavelets, resample, hanning, chirp
from scipy.fftpack import fft, ifft
import numpy as np
from scikits import audiolab
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
    return np.abs(convolved) * hanning_window

def stream_wavelets(signal):
    sig = np.concatenate([np.zeros((M/2,)),
                          signal,
                          np.zeros((M,))])
    lastchunk = np.zeros((nfilters, M/2))
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

if __name__ == '__main__':
    import pylab, time
    #sndfile = audiolab.Sndfile('settler.ogg')
    #signal = np.mean(sndfile.read_frames(44100*20), axis=1)
    signal = chirp(np.arange(2**18), 16.3516/44100, 2**18, 4185.01/44100,
                   method='logarithmic')
    pylab.figure(2)
    pylab.specgram(signal, NFFT=16384, noverlap=16384-4096, Fs=44100)
    pylab.ylim(0, 1000)
    pieces = []
    for piece in stream_wavelets(signal):
        print time.time()
        pieces.append(piece)

    wavelet_graph = np.log(np.concatenate(pieces[1:-2], axis=-1))
    pylab.figure(1)
    pylab.imshow(wavelet_graph[:, ::20], aspect='auto', origin='lower')
    pylab.show()

#mfcc = np.abs(fft(np.log(output[16:112].T)))
#pylab.figure(2)
#pylab.imshow(mfcc[::10, 1:65].T, aspect='auto', origin='lower')

