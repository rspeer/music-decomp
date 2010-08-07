from scipy.signal import fftconvolve, wavelets, resample
from scipy.fftpack import fft, dct
import numpy as np
from scikits import audiolab
RATE = 44100

#def downsample(signal, factor):
#    return resample(signal, signal.shape[-1]/factor)

C0 = 8.1758
M = 32768
def morlet_freq(f0, M):
    """
    Get a Morlet wavelet tuned to detect a particular frequency.
    """
    # this version scales the length of the waveform
    # return wavelets.morlet(RATE*40/f0, 40, 0.5)

    w = wavelets.morlet(M, 40, float(f0*M)/(RATE*80))
    return w / np.linalg.norm(w)

filters = np.zeros((128, M), dtype='complex128')
for x in xrange(128):
    filters[x] = morlet_freq(C0 * 2.0**(x/12.0), M)
print "made filters"

def wavelet_detect(signal, wavelet):
    return np.abs(fftconvolve(signal, wavelet, mode='same'))

def music_wavelets(signal):
    print "building output matrix"
    output = np.zeros((len(filters), len(signal)))
    for band in xrange(len(filters)):
        print band
        output[band] = np.abs(wavelet_detect(signal, filters[band]))
    return output

if __name__ == '__main__':
    import pylab
    sndfile = audiolab.Sndfile('clocks.ogg')
    signal = np.mean(sndfile.read_frames(44100*20), axis=1)
    output = music_wavelets(signal)
    pylab.figure(1)
    pylab.imshow(output[:, ::10], aspect='auto', origin='lower')
    mfcc = dct(np.log(output[16:112, :].T + 0.1))
    pylab.hot()
    pylab.savefig('clocks1.png')
    pylab.figure(2)
    pylab.imshow(mfcc[:, 1:65].T, aspect='auto', origin='lower')
    pylab.savefig('clocks2.png')
