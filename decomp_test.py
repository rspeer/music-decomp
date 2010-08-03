from pyechonest.track import track_from_filename
from pyechonest import config
from scikits import audiolab
from csc import divisi2
from csc.divisi2.export_svdview import write_packed
import numpy as np
import os

config.ECHO_NEST_API_KEY="LFYSHIOM0NNSDBCKJ"

def normalize_dense_cols(array):
    array = array.copy()
    for col in xrange(array.shape[1]):
        array[:,col] /= np.linalg.norm(array[:,col]) + 0.00001
    return array

def analyze_song(filename):
    sndfile = audiolab.Sndfile(filename)
    snddata = sndfile.read_frames(sndfile.nframes)
    rate = sndfile.samplerate

    track = track_from_filename(filename)
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
        next_bar = beats[min(beatnum+meter*2, len(beats)-1)]
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

    diff = songdata[1:] - songdata[:-1]
    msq_diff = np.sqrt((diff ** 2).mean(axis=0))
    filebase = filename[:filename.rfind('.')]
    labels = [filebase+':'+str(i) for i in xrange(len(songdata))]
    songmatrix = divisi2.DenseMatrix(songdata, row_labels=labels)
    return songmatrix, msq_diff

def multi_song_analyze():
    song1, msq1 = analyze_song('clocks.ogg')
    song2, msq2 = analyze_song('settler.ogg')
    song3, msq3 = analyze_song('high-hopes.ogg')
    songmatrix = song1.concatenate(song2).concatenate(song3)
    rms_diff = np.sqrt((msq1 + msq2 + msq3) / 3)
    songmatrix_centered, means = (songmatrix/rms_diff).col_mean_center()
    U, S, V = songmatrix_centered.svd(k=10)
    write_packed(U, os.path.expanduser('~/Documents/Processing/musicsvd/data/combined'))
    return songmatrix_centered, U, S, V

def test():
    return multi_song_analyze()

if __name__ == '__main__':
    songmatrix, U, S, V = test()
