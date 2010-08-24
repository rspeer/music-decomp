from music_theory import HARMONIC_VALUES
import numpy as np
def harmonic_prior(npitches, nperc, nharmonic, nfree):
    N = nharmonic + nperc + nfree
    prior = np.zeros((npitches, N))
    prior[:, :nperc] = 0.00001
    prior[:, nperc:] = -0.00001
    for harmonic, steps in HARMONIC_VALUES:
        prior[steps, nperc:] = 0.001/harmonic
    for harmonic, steps in HARMONIC_VALUES[1:]:
        prior[steps, nperc+nharmonic:] = 0
    return prior

def attack_release(nsteps, a, r, power=1.0):
    curve = np.zeros((nsteps,))
    curve[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    curve[a:a+r] = np.linspace(1.0, 0.0, r, endpoint=False)
    return curve ** power

def adsr(nsteps, a, d, s, st, r):
    curve = np.zeros((nsteps,))
    curve[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    curve[a:a+d] = np.linspace(1.0, s, d, endpoint=False)
    curve[a+d:a+d+st] = s
    curve[a+d+st:a+d+st+r] = np.linspace(s, 0.0, r, endpoint=False)
    return curve

envelope_prior = np.vstack([
    attack_release(30, 1, 5, 2.0),
    attack_release(30, 2, 10, 1.4),
    attack_release(30, 3, 15, 1.0),
    attack_release(30, 4, 20, 0.7),
    attack_release(30, 5, 25, 0.5)
])

