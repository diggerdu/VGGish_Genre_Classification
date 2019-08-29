import librosa
import os.path as osp
import numpy as np
from glob import glob
from multiprocessing import Pool


sr = 44100
hop_len = 512
group_size = 10
patch_size = 50
class_num = 62
group_len = group_size * patch_size


def feat_extract(audio_file):
    print(audio_file)
    # note_file = '../../audio_data/changba/changba_midi_adjust/%s' % (
    #         osp.basename(audio_file).replace('_ktv3url.mp3', '.note'))
    # if not osp.exists(note_file):
    #     return
    __import__('pdb').set_trace()     
    audio, _ = librosa.load(audio_file, sr=sr, mono=True)
    audio /= np.std(audio)
    audio -= np.mean(audio)

    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_len,
            fmin=55.0, n_bins=301, bins_per_octave=60)
    cqt = np.abs(cqt)

    n_groups = int(np.ceil(cqt.shape[1] / group_len))
    n_frames = n_groups * group_len
    
    feat = np.hstack((cqt, np.zeros((cqt.shape[0], n_frames - cqt.shape[1]))))

    # feat = np.hsplit(feat, n_groups)
    # feat = np.array([np.hsplit(group, group_size) for group in feat])
    feat = np.array(np.hsplit(feat, n_groups))
    feat = np.array([np.hsplit(feat[i, :, :], group_size) for i in range(feat.shape[0])])
    feat = np.moveaxis(feat, -1, -2)

    return feat, cqt.shape[1]
    
    '''    
    # note_file = audio_file.replace('.mp3', '.note')
    note_mat = np.loadtxt(note_file)

    pitch_time = np.arange(n_frames) * hop_len / sr
    pitch_ref = np.ones(n_frames, dtype=int) * 61
    for i in range(note_mat.shape[0]):
        t0 = note_mat[i, 0] / 1000.0
        t1 = (note_mat[i, 0] + note_mat[i, 1]) / 1000.0
        
        pv = int(note_mat[i, 2]) - 30
        if pv >= 61:
            pv = 60
        elif pv < 0:
            pv = 61

        pitch_ref[(pitch_time >= t0) & (pitch_time <= t1)] = pv
    
    label = np.zeros((class_num, n_frames), dtype=int)
    for i, pv in enumerate(pitch_ref):
        label[pv, i] = 1

    label = np.array(np.hsplit(label, n_groups), dtype=int)
    label = np.moveaxis(label, -1, -2)

    feat_file = 'changba_feat/%s' % (
            osp.basename(audio_file).replace('_ktv3url.mp3', ''))
    np.save(feat_file + '.feat', feat)
    np.save(feat_file + '.lab', label)
    '''


if __name__ == '__main__':
    feat_extract("/root/thome/data/lab_aligned_data/3302.mp3")
    audio_file_list = glob('../../audio_data/changba/changba_audio/*.mp3')
    
    p = Pool(20)
    p.map(feat_extract, audio_file_list)
    p.close()
    p.join()
