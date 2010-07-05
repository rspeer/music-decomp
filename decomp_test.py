from pyechonest.track import Track
from pyechonest import config
from scikits import audiolab
from csc import divisi2
import numpy as np

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
    segments = track.segments
    songdata = np.zeros((len(bars)-1, 26))

    for barnum in xrange(len(bars)-1):
        start = bars[barnum]
        end = bars[barnum+1]
        segs = [seg for seg in track.segments if seg['start'] >= start and seg['start'] < end]
        if segs:
            pitch = np.zeros((12,))
            timbre = np.zeros((12,))
            loudness = 0
            for seg in segs:
                pitch += np.array(seg['pitches'])
                timbre += np.array(seg['timbre'])
                loudness += seg['loudness_max']
            pitch /= len(segs)
            timbre /= len(segs)
            loudness /= len(segs)
            segrate = len(segs)
            
            songdata[barnum,0:12] = pitch
            songdata[barnum,12:24] = timbre
            songdata[barnum,24] = loudness
            songdata[barnum,25] = segrate

        #start_samp = int(bars[barnum] * rate)
        #end_samp = int(bars[barnum+1] * rate)

        #audiolab.play(snddata[start_samp:end_samp].T)

    divisi_song = divisi2.DenseMatrix(songdata)
    divisi2.save(divisi_song, TEST_FILENAME+'.pickle')
    return normalize_dense_cols(divisi_song)

songdata = test()

