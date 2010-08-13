from scipy.signal import fftconvolve, wavelets, resample, hanning, chirp, square, sawtooth
from scipy.fftpack import fft, ifft
import numpy as np
from scikits import audiolab
from csc import divisi2
RATE = 44100

A0 = 27.5
M = 65536
nfilters = 96
EPS = 1e-8
ALPHA = 0.5
SUBSAMPLE = 256

HARMONIC_VALUES = [(1, 0),
                   (2, 12),
                   (3, 19),
                   (4, 24),
                   (5, 28),
                   (6, 31),
                   (8, 36),
                   (9, 38),
                   (10, 40),
                   (12, 43),
                   (15, 47),
                   (16, 48),
                   (18, 50),
                   (20, 52),
                   (24, 55),
                   (27, 57),
                   (30, 59),
                   (32, 60)]

NOISE_SHAPE = hanning(25) / np.sum(hanning(25)) * 3.1

def triangle(signal):
    return np.abs(sawtooth(signal)) - 0.5

def morlet_freq(f0, M):
    """
    Get a Morlet wavelet tuned to detect a particular frequency.
    """
    # this version scales the length of the waveform
    # return wavelets.morlet(RATE*40/f0, 40, 0.5)

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    return w * (np.sqrt(f0)/RATE) / np.linalg.norm(w)

def morlet_freq_harmonic(f0, M):
    # this doesn't actually work

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    for harmonic in xrange(2, 9):
        w += wavelets.morlet(M, 40, float(f0*M*harmonic)/(RATE*80))/harmonic
        w -= wavelets.morlet(M, 40, float(f0*M)/(harmonic*RATE*80))/harmonic
    return w / np.linalg.norm(w)

def downsample(signal, factor):
    return resample(signal, signal.shape[-1]/factor)

fftfilters = np.zeros((nfilters, M*2), dtype='complex128')
for x in xrange(nfilters):
    filter1 = morlet_freq(A0 * 2.0**(x/12.0), M)
    fftfilters[x, :M] = fft(filter1)
hanning_window = hanning(M)
print "made filters"

def wavelet_detect(signal):
    """
    Uses a set of wavelet filters to detect pitches in the signal, with
    constant pitch resolution (higher than a typical STFT!)
    
    The inevitable tradeoff comes in the form of time resolution in the bass
    notes.
    """
    fftsignal = fft(signal * hanning_window).conj()
    fftsignal = np.concatenate([fftsignal, np.zeros(fftsignal.shape)], axis=-1)
    #convolved = ifft(fftfilters * fftsignal)[:, M-1::-1]
    convolved = np.roll(ifft(fftfilters * fftsignal), M, axis=-1)[:, ::-2]
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

def detect_harmonics(matrix, shape):
    F, T = matrix.shape
    alignment = np.zeros((F, T))
    matrix = np.concatenate([matrix, np.ones((60, T))*np.mean(matrix)]) + EPS
    lmatrix = np.log(matrix)
    if shape == 'sawtooth':
        for h, steps in HARMONIC_VALUES:
            alignment += lmatrix[steps:steps+F] * 1.0/h
    elif shape == 'triangle':
        for h, steps in HARMONIC_VALUES:
            if h % 2 == 1:
                alignment += lmatrix[steps:steps+F] * 3.2/h/h
    else:
        raise NotImplementedError

    deharmonized = np.exp(alignment)
    return deharmonized

def detect_noise(matrix):
    F, T = matrix.shape
    offset = len(NOISE_SHAPE)//2
    alignment = np.zeros((F, T))
    matrix = np.concatenate([np.ones((offset, T)) * np.mean(matrix), matrix, np.ones((offset, T)) * np.mean(matrix)]) + EPS
    lmatrix = np.log(matrix)

    for i, mag in enumerate(NOISE_SHAPE):
        alignment += lmatrix[i:i+F] * mag

    noise_profile = np.exp(alignment)
    return noise_profile

meansq = EPS
def timbre_color(matrix):
    # okay, this really is the evil kind of global var. I'll classify sometime
    global meansq
    amplitude = np.sum(matrix, axis=0)
    rgb = np.zeros(matrix.shape + (3,))
    rgb[:,:,0] = detect_noise(matrix)
    rgb[:,:,1] = detect_harmonics(matrix, 'sawtooth')
    rgb[:,:,2] = detect_harmonics(matrix, 'triangle')
    
    amp_adjust = np.sum(np.sum(rgb, axis=2), axis=0) / (amplitude+EPS)
    rgb /= amp_adjust[np.newaxis, :, np.newaxis]

    prev_meansq = meansq
    meansq = (meansq*(1.0-ALPHA)) + (np.mean(rgb*rgb) * ALPHA)
    meansq_smooth = np.linspace(np.sqrt(prev_meansq), np.sqrt(meansq), matrix.shape[-1])
    return np.minimum(rgb/meansq_smooth[np.newaxis, :, np.newaxis]/10, 1)

def reconstruct(matrix):
    pcm = np.zeros((matrix.shape[1]*SUBSAMPLE,))
    angle = np.arange(matrix.shape[1]*SUBSAMPLE) * 2 * np.pi / RATE
    for pitch in xrange(nfilters):
        print pitch
        freq = A0 * 2.0**(pitch/12.0)
        square_wave = square(angle*freq)
        triangle_wave = triangle(angle*freq)
        pcm += np.repeat(matrix[pitch, :, 1], SUBSAMPLE) * square_wave
        pcm += np.repeat(matrix[pitch, :, 2], SUBSAMPLE) * triangle_wave
    pcm /= np.max(pcm)
    return pcm

def smooth(matrix, n=20):
    end = matrix.shape[1]-n
    result = matrix[:, 0:end]
    for i in xrange(n):
        result = np.maximum(result, matrix[:, i:end+i])
    return result

if __name__ == '__main__':
    import pylab, time
    sndfile = audiolab.Sndfile('blinding_lights.ogg')
    signal = np.mean(sndfile.read_frames(44100*60), axis=1)
    #signal = chirp(np.arange(2**18), 16.3516/44100, 2**18, 4185.01/44100,
    #               method='logarithmic')
    pieces = []
    pcmpieces = []
    for piece in stream_wavelets(signal):
        print time.time()
        piece = timbre_color(np.abs(piece[:, ::SUBSAMPLE]))
        pieces.append(piece)

    wavelet_graph = np.concatenate(pieces, axis=1)
    pcm = reconstruct(wavelet_graph)
    audiolab.play(pcm)
    
    pylab.figure(1)
    pylab.imshow(wavelet_graph, aspect='auto', origin='lower')
    pylab.show()

#mfcc = np.abs(fft(np.log(output[16:112].T)))
#pylab.figure(2)
#pylab.imshow(mfcc[::10, 1:65].T, aspect='auto', origin='lower')

