from scikits.talkbox.features import trfbank
from scipy.signal import hamming, lfilter3
import numpy as np
import math

A0 = 27.5
HALF_STEP = math.pow(2.0, 1.0/12)
FFTSIZE = 8192  # I think this is the number of FFT frequencies
harmonics = [0, 12, 19, 24, 28, 31, 34, 36, 38, 40]
# Make a triangular filterbank corresponding to musical notes
music_filterbank = trfbank(
    fs       = 12000.0,   # your guess is as good as mine
    nfft     = FFTSIZE,
    lowfreq  = A0,        # begin with the lowest key on a piano
    linsc    = 1.0,       # irrelevant
    logsc    = HALF_STEP, # take half-step increments from there
    nlinfilt = 0,         # no linear steps
    nlogfilt = 128        # 128 pitch steps

    # with 128 pitch steps, we can get to the 10th harmonic of the top note
    # on a piano!
)

def music_spectrum(segs):
    """
    Find the equal-tempered musical spectrum of a signal chopped up into
    equal-sized segments. Returns the log-power of 128 different pitches
    starting from A0.
    """
    #segs = np.atleast_2d(segs)
    width = segs.shape[-1]
    hamming_window = hamming(width, sym=False)
    spectrum = np.abs(fft(segs, FFTSIZE, axis=-1))
    mspec = np.log(np.dot(spectrum, music_filterbank.T))
    return mspec
