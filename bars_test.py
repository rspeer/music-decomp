from pyechonest.track import Track
from pyechonest import config
from scikits import audiolab
from csc import divisi2
import mad
import numpy as np

config.ECHO_NEST_API_KEY="LFYSHIOM0NNSDBCKJ"

TEST_FILENAME = "settler.ogg"

def test():
    sndfile = audiolab.Sndfile("settler.ogg")
    snddata = sndfile.read_frames(sndfile.nframes)
    rate = sndfile.samplerate

    track = Track(TEST_FILENAME)
    bars = [bar['start'] for bar in track.bars]
    segments = track.segments
    print segments

    for barnum in xrange(len(bars)-1):
        start = bars[barnum]
        end = bars[barnum+1]
        segs = [seg for seg in track.segments if seg['start'] >= start and seg['start'] < end]
        pitch = np.zeros((12,))
        timbre = np.zeros((12,))
        loudness = 0
        for seg in segs:
            pitch += np.ndarray(seg['pitch'])
            timbre += np.ndarray(seg['timbre'])
            loudness += seg['loudness_max']
        pitch /= len(segs)
        timbre /= len(segs)
        loudness /= len(segs)


        start_samp = int(bars[barnum] * rate)
        end_samp = int(bars[barnum+1] * rate)

        audiolab.play(snddata[start_samp:end_samp].T)


if __name__ == '__main__':
    test()

