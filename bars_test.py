from pyechonest.track import Track
from pyechonest import config
from scikits import audiolab
from csc import divisi2
import numpy as np

config.ECHO_NEST_API_KEY="LFYSHIOM0NNSDBCKJ"

TEST_FILENAME = "clocks.ogg"

def test():
    sndfile = audiolab.Sndfile(TEST_FILENAME)
    snddata = sndfile.read_frames(sndfile.nframes)
    rate = sndfile.samplerate

    track = Track(TEST_FILENAME)
    bars = [bar['start'] for bar in track.bars]

    for barnum in xrange(len(bars)-1):
        start = bars[barnum]
        end = bars[barnum+1]

        start_samp = int(bars[barnum] * rate)
        end_samp = int(bars[barnum+1] * rate)

        audiolab.play(snddata[start_samp:end_samp].T)

if __name__ == '__main__':
    test()

