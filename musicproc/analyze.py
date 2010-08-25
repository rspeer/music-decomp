import numpy as np
#from matplotlib import pyplot as plt
import pylab as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from scikits import audiolab
from music_theory import HARMONIC_VALUES

try:
    from pyechonest.track import track_from_filename
except ImportError:
    from pyechonest.track import Track as track_from_filename

from pyechonest import config
config.ECHO_NEST_API_KEY="LFYSHIOM0NNSDBCKJ"
import time

def triangle(sig):
    """
    Make a triangle wave from a signal (in radians), similarly to signal.square
    and signal.sawtooth.
    """
    return np.abs(signal.sawtooth(sig)) - 0.5

class AudioData(object):
    def __init__(self, sig, rate, filename=None):
        self.signal = sig
        self.rate = rate
        self.filename = filename
    
    @staticmethod
    def from_file(filename):
        sndfile = audiolab.Sndfile(filename)
        rate = sndfile.samplerate
        sig = sndfile.read_frames(sndfile.nframes)
        return AudioData(sig, rate, filename)

    def get_mono(self, start, end=None):
        startsamp = start*self.rate
        if end is None:
            endsamp = len(self.signal)
        else:
            endsamp = end*self.rate
        return np.mean(self.signal[startsamp:endsamp], axis=1)

    def get_echonest(self):
        return track_from_filename(self.filename)

    def downsample(self, factor):
        sig = signal.resample(self.signal, len(self.signal)/factor, axis=0,
                              window='hanning')
        return AudioData(sig, self.rate/factor)

    def length(self):
        return len(self.signal)/self.rate

    def play(self):
        audiolab.play(self.signal.T)

class MusicAnalyzer(object):
    def __init__(self, lowfreq=27.5, npitches=96, samplerate=44100,
                 window_size=65536, subsample=256, epsilon=1e-8):
        self.lowfreq = lowfreq
        self.npitches = npitches
        self.samplerate = samplerate
        self.window_size = window_size
        self.subsample = subsample
        self.epsilon = epsilon
        self.filterbank = MorletFilterBank(lowfreq, npitches,
                                           self.window_size, self.samplerate)
        #self.filter_window = signal.hanning(self.window_size)
        self.filter_window = np.ones((self.window_size,))
        self.timbre_analyzer = BasicTimbreAnalyzer()
    
    def stream_analyze_pitch(self, audio, maxlen):
        # TODO: accept an actual stream as input
        assert audio.rate == self.samplerate
        sig = np.concatenate([np.zeros((self.window_size/2,)),
                              audio.get_mono(0, maxlen),
                              np.zeros((self.window_size,))])
        lastchunk = np.zeros((self.npitches, self.window_size/2), dtype='complex128')
        
        M = self.window_size
        skip = True
        for frame in xrange(1, sig.shape[-1]*2/M - 1):
            chunk = self.filterbank.analyze(sig[frame*M/2 : (frame+2)*M/2],
                                            window=self.filter_window)
            if skip:
                skip = False
            else:
                yield (lastchunk + chunk[:, :M/2])
            lastchunk[:] = chunk[:, M/2:]
    
    def stream_analyze_timbre(self, audio, maxlen):
        # TODO: quantize
        for chunk in self.stream_analyze_pitch(audio, maxlen):
            print time.time()
            mod_chunk = np.abs(chunk)[:, ::self.subsample]
            yield self.timbre_analyzer.analyze(mod_chunk)

    def analyze_pitch(self, audio, maxlen=60):
        pitch_chunks = []
        for pitch in self.stream_analyze_pitch(audio, maxlen):
            pitch_chunks.append(pitch)
        return np.concatenate(pitch_chunks, axis=1)

    def analyze_timbre(self, audio, maxlen=60):
        timbre_chunks = []
        for timbre in self.stream_analyze_timbre(audio, maxlen):
            timbre_chunks.append(timbre)
        return np.concatenate(timbre_chunks, axis=1)

    def live_plot(self, audio, maxlen=None):
        timbre_chunks = [np.zeros((self.npitches, self.window_size // self.subsample, 3)) for i in xrange(20)]
        counter = 0
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        fig.set_dpi(100)
        fig.set_size_inches(self.window_size // self.subsample * 20 / 100.0,
                                 self.npitches * 4 / 100.0)
        #self.outfile = open('timbre.dat', 'wb')
                            
        for timbre in self.stream_analyze_timbre(audio, maxlen):
            timbre_chunks.append(timbre)
            timbre_chunks = timbre_chunks[-20:]
            timbre_all = np.concatenate(timbre_chunks, axis=1)
            #self.save_data(timbre)
            self.plot(timbre_all)
            counter += 1
        #self.outfile.close()
        return timbre_all
    
    def plot(self, timbre):
        plt.imshow(np.minimum(1.0, timbre*2), aspect='auto',
                   origin='lower', interpolation='nearest')
        plt.draw()

    def save_data(self, timbre):
        bytes = np.array(np.minimum(255, timbre*512), dtype=np.uint8)
        for i in xrange(bytes.shape[1]):
            self.outfile.write(bytes[:,i].tostring())


    def get_tatum_timeline(self, audio):
        track = audio.get_echonest()
        times = []
        for tatum in track.tatums:
            times.append(tatum['start'])
        times.append(audio.length())
        return times
    
    def quantize_subsampled(self, data, timeline):
        return self.quantize(data, timeline, self.samplerate / self.subsample)

    def quantize(self, data, timeline, rate):
        output = np.zeros(data.shape)
        for timestep in xrange(len(timeline)-1):
            start = timeline[timestep]*rate
            end = timeline[timestep+1]*rate
            if start > output.shape[1]: break
            output[:, start:end] = np.mean(data[:, start:end], axis=1)[:, np.newaxis]
        return output
    
    def quantize_equal(self, data, width):
        return self.quantize(data, np.arange(data.shape[1])[::width], 1)[:, ::width]

    def reconstruct(self, timbre):
        pcm = np.zeros((timbre.shape[1]*self.subsample,))
        angle = np.arange(timbre.shape[1]*self.subsample) * 2 * np.pi / self.samplerate
        for pitch in xrange(self.npitches):
            print pitch
            freq = self.lowfreq * 2.0**(pitch/12.0)
            square_wave = signal.square(angle*freq)
            triangle_wave = triangle(angle*freq)
            pcm += np.repeat(timbre[pitch, :, 1], self.subsample) * square_wave
            pcm += np.repeat(timbre[pitch, :, 2], self.subsample) * triangle_wave
        pcm /= np.max(pcm)
        return AudioData(pcm, self.samplerate)

    def reconstruct_W(self, W):
        pcm = np.zeros((W.shape[1]*self.subsample,))
        angle = np.arange(W.shape[1]*self.subsample) * 2 * np.pi / self.samplerate
        for pitch in xrange(self.npitches):
            print pitch
            freq = self.lowfreq * 2.0**(pitch/12.0)
            sine = np.sin(angle*freq)
            pcm += np.repeat(W[pitch], self.subsample) * sine
        pcm /= np.max(pcm)
        return AudioData(pcm, self.samplerate)
        
    def reconstruct_WZH(self, plca, W, Z, H):
        WZH = plca.reconstruct(W, Z, H, circular=[False, False])
        pcm = np.zeros((WZH.shape[1]*self.subsample,))
        angle = np.arange(WZH.shape[1]*self.subsample) * 2 * np.pi / self.samplerate
        for pitch in xrange(self.npitches):
            print pitch
            freq = self.lowfreq * 2.0**(pitch/12.0)
            sine = np.sin(angle*freq)
            pcm += np.repeat(WZH[pitch], self.subsample) * sine
        pcm /= np.max(pcm)
        return AudioData(pcm, self.samplerate)

class MorletFilterBank(object):
    def __init__(self, lowfreq, npitches, window_size, samplerate,
                 freq_resolution=40):
        self.samplerate = samplerate
        self.npitches = npitches
        self.window_size = window_size
        self.lowfreq = lowfreq
        self.freq_resolution = freq_resolution
        self.filters = self.make_filters()

    def __len__(self):
        return self.npitches

    def make_morlet(self, freq):
        """
        Get a Morlet wavelet tuned to detect a particular frequency.
        """
        w = signal.wavelets.morlet(self.window_size, self.freq_resolution,
                                   float(freq*self.window_size)
                                    /(self.samplerate*self.freq_resolution*2))
        return w * np.sqrt(freq / self.samplerate) / np.linalg.norm(w)

    def pitch_to_freq(self, p):
        return self.lowfreq * 2.0**(p/12.0)
        
    def make_filters(self):
        filters = np.zeros((self.npitches, self.window_size*2),
                           dtype='complex128')
        for p in xrange(self.npitches):
            morlet = self.make_morlet(self.pitch_to_freq(p))
            filters[p, :self.window_size] = fft(morlet)
        return filters

    def analyze(self, sig, window):
        fftsignal = fft(sig * window).conj()
        fftsignal = np.concatenate([fftsignal, np.zeros(fftsignal.shape)],
                                   axis=-1)
        convolved = np.roll(ifft(self.filters * fftsignal), self.window_size, axis=-1)[:, ::-2]
        return convolved * window

class BasicTimbreAnalyzer(object):
    def __init__(self, sawtooth_scale=1.0, triangle_scale=3.2, noise_scale=3.1, epsilon=1e-8):
        # Until I can compute a legitimate prior, these will have to do.
        self.sawtooth_scale = sawtooth_scale
        self.triangle_scale = triangle_scale
        self.noise_scale = noise_scale
        self.noise_shape = signal.hanning(25) / np.sum(signal.hanning(25)) * self.noise_scale
        self.epsilon = epsilon
        self.meansq = self.epsilon
        self.smoothing = 0.5

    def detect_harmonics(self, matrix, shape):
        F, T = matrix.shape
        alignment = np.zeros((F, T))
        matrix = np.concatenate([matrix, np.ones((60, T))*np.mean(matrix)]) + self.epsilon
        lmatrix = np.log(matrix)
        if shape == 'sawtooth':
            for h, steps in HARMONIC_VALUES:
                alignment += lmatrix[steps:steps+F] * self.sawtooth_scale/h
        elif shape == 'triangle':
            for h, steps in HARMONIC_VALUES:
                if h % 2 == 1:
                    alignment += lmatrix[steps:steps+F] * self.triangle_scale/h/h
        else:
            raise NotImplementedError

        deharmonized = np.exp(alignment)
        return deharmonized

    def detect_noise(self, matrix):
        F, T = matrix.shape
        offset = len(self.noise_shape)//2
        alignment = np.zeros((F, T))
        matrix = np.concatenate([np.ones((offset, T)) * np.mean(matrix), matrix, np.ones((offset, T)) * np.mean(matrix)]) + self.epsilon
        lmatrix = np.log(matrix)

        for i, mag in enumerate(self.noise_shape):
            alignment += lmatrix[i:i+F] * mag

        noise_profile = np.exp(alignment)
        return noise_profile

    def analyze(self, matrix):
        amplitude = np.sum(matrix, axis=0)
        rgb = np.zeros(matrix.shape + (3,))
        rgb[:,:,0] = self.detect_noise(matrix)
        rgb[:,:,1] = self.detect_harmonics(matrix, 'sawtooth')
        rgb[:,:,2] = self.detect_harmonics(matrix, 'triangle')
        
        amp_adjust = np.sum(np.sum(rgb, axis=2), axis=0) / (amplitude+self.epsilon)
        rgb /= amp_adjust[np.newaxis, :, np.newaxis]

        prev_meansq = self.meansq
        self.meansq = (self.meansq*(1.0-self.smoothing)) + (np.mean(rgb*rgb) * self.smoothing)
        meansq_smooth = np.linspace(np.sqrt(prev_meansq), np.sqrt(self.meansq),
                                    matrix.shape[-1])
        return np.minimum(rgb/meansq_smooth[np.newaxis, :, np.newaxis]/2, 1)

if __name__ == '__main__':
    audio = AudioData.from_file('../chess.ogg')
    analyzer = MusicAnalyzer(window_size=44100, subsample=1470)
    #plt.plot(analyzer.filter_window)
    #plt.show()
    pitch = analyzer.quantize_equal(np.abs(analyzer.analyze_pitch(audio, 15)), 1470)

