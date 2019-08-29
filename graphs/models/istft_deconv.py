import torch
import torch.nn.functional as F


def istft(stft_matrix,length, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True):
    """stft_matrix = (freq, time, 2) (batch dimension not included)
    - Based on librosa implementation and Keunwoo Choi's implementation
        - librosa: http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
        - Keunwoo Choi's: https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e#file-istft-torch-py
    """
    assert normalized == False
    assert onesided == True
    assert window == 'hann'
    assert center == True
        
    __import__('pdb').set_trace() 
    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft, device=device)

    n_frames = stft_matrix.shape[1]
    expected_signal_len =  n_fft + hop_length * (n_frames - 1)

    conj = torch.tensor([1., -1.], requires_grad=False, device=device)

    # [a,b,c,d,e] -> [a,b,c,d,e,d,c,b]
    stft_matrix = torch.cat(
        (stft_matrix, conj*stft_matrix.flip(dims=(0,))[1:-1]), dim=0)
    # now shape is [n_fft, T, 2]

    stft_matrix = stft_matrix.transpose(0, 1)
    stft_matrix = torch.ifft(stft_matrix, signal_ndim=1)[:, :, 0] # get real part of ifft
    ytmp = stft_matrix * istft_window
    ytmp = ytmp.transpose(0, 1)
    ytmp = ytmp.unsqueeze(0)
    # now [1, n_fft, T]. this is stack of `ytmp` in librosa/core/spectrum.py

    eye = torch.eye(n_fft, requires_grad=False, device=device)
    eye = eye.unsqueeze(1) # [n_fft, 1, n_fft]

    y = F.conv_transpose1d(ytmp, eye, stride=hop_length, padding=0)
    y = y.view(-1)
    assert y.size(0) == expected_signal_len

    y = y[n_fft//2:]
    y = y[:length]
    coeff = n_fft/float(hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff
