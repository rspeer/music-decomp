from pyechonest.track import Track
from pyechonest import config
from scikits import audiolab
from csc import divisi2
import numpy as np
from pylab import *

config.ECHO_NEST_API_KEY="LFYSHIOM0NNSDBCKJ"

TEST_FILENAME = "settler.ogg"

def normalize_dense_cols(array):
    array = array.copy()
    for col in xrange(array.shape[1]):
        array[:,col] /= np.linalg.norm(array[:,col]) + 0.00001
    return array

def test():
    sndfile = audiolab.Sndfile(TEST_FILENAME)
    snddata = sndfile.read_frames(sndfile.nframes)
    rate = sndfile.samplerate

    track = Track(TEST_FILENAME)
    bars = [bar['start'] for bar in track.bars]
    beats = [beat['start'] for beat in track.beats]
    meter = float(len(beats))/len(bars)
    print meter
    meter = int(np.round(meter))

    segments = track.segments
    songdata = np.zeros((len(beats)-1, 27))

    for beatnum in xrange(len(beats)-1):
        start = beats[beatnum]
        end = beats[beatnum+1]
        next_bar = beats[min(beatnum+meter, len(beats)-1)]
        segs = [seg for seg in track.segments if seg['start'] >= start and seg['start'] < end]
        bar_segs = [seg for seg in track.segments if seg['start'] >= start and seg['start'] < next_bar]
        bpm = 60.0/(end-start)
        if segs or bar_segs:
            pitch = np.zeros((12,))
            timbre = np.zeros((12,))
            loudness = 0
            for seg in segs:
                pitch += np.array(seg['pitches'])
                timbre += np.array(seg['timbre'])
                loudness += seg['loudness_max']
            for seg in bar_segs:
                pitch += np.array(seg['pitches'])
                timbre += np.array(seg['timbre'])
                loudness += seg['loudness_max']
            n = len(segs)+len(bar_segs)
            pitch /= n
            timbre /= n
            loudness /= n
            segrate = len(segs)
            
            songdata[beatnum,0:12] = pitch
            songdata[beatnum,12:24] = timbre
            songdata[beatnum,24] = loudness
            songdata[beatnum,25] = segrate
            songdata[beatnum,26] = bpm

    divisi_song = divisi2.DenseMatrix(songdata)
    divisi2.save(divisi_song, TEST_FILENAME+'.pickle')
    return (snddata, beats, rate, normalize_dense_cols(divisi_song))

snddata, beats, rate, mat = test()
U, S, V = mat.svd(k=10)
songlen = len(beats)
colors = [(1.0, x/float(songlen), 0.0) for x in xrange(songlen)]
plot(U[:,0], U[:,1], 'k-')
scatter(U[:,0], U[:,1], facecolors=colors, edgecolors=colors)

