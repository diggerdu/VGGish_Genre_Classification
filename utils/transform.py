from librosa import cqt, icqt
import numpy as np

def gl_cqt(S, n_iter=32, sr=22050, hop_length=512, bins_per_octave=12, fmin=None, window='hann',
               dtype=np.float32, length=None, momentum=0.99, random_state=None, res_type='kaiser_fast'):

    if fmin is None:
        fmin = librosa.note_to_hz('C1')
        
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise ParameterError('griffinlim() called with momentum={} < 0'.format(momentum))

    # randomly initialize the phase
    angles = np.exp(2j * np.pi * rng.rand(*S.shape))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt
        __import__('pdb').set_trace()  
        # Invert with our current estimate of the phases
        inverse = icqt(S * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, fmin=fmin,
                        #window=window, length=length, res_type=res_type)
                        window=window)

        # Rebuild the spectrogram
        rebuilt = cqt(inverse, sr=sr, bins_per_octave=bins_per_octave, n_bins=S.shape[0],
                       hop_length=hop_length, fmin=fmin,
                       window=window, res_type=res_type)

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    # Return the final phase estimates
    return icqt(S * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, fmin=fmin,
                  window=window,length=length, res_type=res_type)
