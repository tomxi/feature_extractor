import os
from tqdm import tqdm
import argparse
import numpy as np

import librosa
import crema
import openl3
import tensorflow_hub as hub


def extract_openl3_yamnet(
    audio_dir = '/scratch/qx244/data/salami/audio/10/', 
    feature_dir = '/scratch/qx244/data/salami/script_out/',
    recompute = False,
):
    # create output directory if not already there.
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
    feat_paths = []
    
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    print('extracting openl3 and yamnet features')
    for track in tqdm(audio_paths):
        audio_path = os.path.join(audio_dir, track)
        track_bn = os.path.basename(audio_path).split('.')[0]
        
        #YAMNET
        yamnet_path = os.path.join(feature_dir, f'{track_bn}_yamnet.npz')
        if recompute or not os.path.exists(yamnet_path):
            yamnet_audio, yamnet_sr = librosa.load(track, sr=16000)
            _, yamnet_emb, _ = yamnet_model(yamnet_audio)
            resampled_yamnet_emb = librosa.resample(
                yamnet_emb.numpy().T,
                orig_sr=1/0.48, target_sr=22050/4096
            )
            yamnet_ts = librosa.frames_to_time(
                np.arange(resampled_yamnet_emb.shape[-1]), 
                sr=22050, hop_length=4096)
            np.savez(yamnet_path, feature=resampled_yamnet_emb, ts=yamnet_ts)

        feat_paths.append(yamnet_path)
        
        #OPENL3
        openl3_path = os.path.join(feature_dir, f'{track_bn}_openl3.npz')
        if recompute or not os.path.exists(openl3_path):
            y, sr = librosa.load(track, sr=22050)
            emb, ts = openl3.get_audio_embedding(
                y, sr, 
                embedding_size=512, content_type='music'
            )
            resampled_emb = librosa.resample(
                emb.T, orig_sr=1 / (ts[1] - ts[0]),
                target_sr= sr/4096
            )
            ts = librosa.frames_to_time(
                np.arange(resampled_emb.shape[-1]), 
                hop_length=4096, sr=sr
            )
            np.savez(openl3_path, feature=resampled_emb, ts=ts)
        feat_paths.append(openl3_path)
        
        
    return feat_paths


def extract_mfcc_tempogram(
    audio_dir = '/scratch/qx244/data/salami/audio/10/', 
    feature_dir = '/scratch/qx244/data/salami/script_out/',
    recompute = False,
):
    # create output directory if not already there.
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    
    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
    feat_paths = []

    print('extracting mfcc and tempogram features')
    for track in tqdm(audio_paths):
        audio_path = os.path.join(audio_dir, track)
        track_bn = os.path.basename(audio_path).split('.')[0]
        y, sr = librosa.load(track, sr=22050)
        
        # MFCC
        mfcc_path = os.path.join(feature_dir, f'{track_bn}_mfcc.npz')
        if recompute or not os.path.exists(mfcc_path):
            # compute mfcc
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=40, 
                hop_length=4096, n_fft=8192, lifter=0.6)
            normalized_mfcc = (mfcc - np.mean(mfcc, axis=1)[:, None]) / np.std(mfcc, axis=1, ddof=1)[:,None]
            mfcc_ts = librosa.frames_to_time(
                np.arange(normalized_mfcc.shape[-1]), 
                hop_length=4096, sr=sr, n_fft=8192)
            np.savez(mfcc_path, feature=normalized_mfcc, ts=mfcc_ts)
        feat_paths.append(mfcc_path)
        
        # TEMPOGRAM
        tempogram_path = os.path.join(feature_dir, f'{track_bn}_tempogram.npz')
        if recompute or not os.path.exists(tempogram_path):
            #compute tempogram
            novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
            tempogram = librosa.feature.tempogram(
                onset_envelope=novelty, sr=sr, 
                hop_length=512, win_length=384)
            resampled_tempogram = librosa.resample(
                tempogram, orig_sr=sr/512, 
                target_sr=sr/4096)
            ts = librosa.frames_to_time(
                np.arange(resampled_tempogram.shape[-1]), 
                hop_length=4096, sr=sr)
            np.savez(tempogram_path, feature=resampled_tempogram, ts=ts)
        feat_paths.append(tempogram_path)

    return feat_paths


def extract_crema(
    audio_dir = '/scratch/qx244/data/salami/audio/10/', 
    feature_dir = '/scratch/qx244/data/salami/script_out/',
    recompute = False,
):
    # create output directory if not already there.
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
      
    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
    feat_paths = []
    
    #load CREMA MODEL
    chord_model = crema.models.chord.ChordModel()
    
    print('extracting crema features')
    for track in tqdm(audio_paths):
        audio_path = os.path.join(audio_dir, track)
        track_bn = os.path.basename(audio_path).split('.')[0]
        
        #CREMA
        crema_path = os.path.join(feature_dir, f'{track_bn}_crema.npz')
        if recompute or not os.path.exists(crema_path):
            # COMPUTE CREMA AND SAVE
            crema_out = chord_model.outputs(track)  
            crema_op = chord_model.pump.ops[2]
            
            resampled_crema_pitch = librosa.resample(
                crema_out['chord_pitch'].T, 
                orig_sr=crema_op.hop_length / crema_op.sr, 
                target_sr=22050/4096
            )
            
            resampled_crema_root = librosa.resample(
                crema_out['chord_root'].T, 
                orig_sr=crema_op.hop_length / crema_op.sr, 
                target_sr=22050/4096
            )
            
            resampled_crema_bass = librosa.resample(
                crema_out['chord_bass'].T, 
                orig_sr = crema_op.hop_length / crema_op.sr, 
                target_sr = 22050/4096
            )

            crema_ts = librosa.frames_to_time(
                np.arange(resampled_crema_pitch.shape[-1]), 
                hop_length=4096, sr=22050)

            np.savez(crema_path, 
                     pitch=resampled_crema_pitch, 
                     root=resampled_crema_root,
                     bass=resampled_crema_bass,
                     ts=crema_ts)
        feat_paths.append(crema_path)
        
    return feat_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Feature Extractor')
    parser.add_argument('audio_dir', help='Path to directory containing audio files')
    parser.add_argument('feature_dir', help='Path to feature output directory')

    parser.add_argument(
        '--recompute', 
        action='store_true', 
        help='Optional Flag to ignore existing computed features in the output directory and recompute everything'
    )
    parser.set_defaults(recompute=False)

    kwargs = parser.parse_args()

    extract_crema(kwargs.audio_dir, kwargs.feature_dir, kwargs.recompute)
    extract_openl3_yamnet(kwargs.audio_dir, kwargs.feature_dir, kwargs.recompute)
    extract_mfcc_tempogram(kwargs.audio_dir, kwargs.feature_dir, kwargs.recompute)

