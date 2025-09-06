from beat_this.inference import Audio2Beats

import argparse
import os
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as tf_hub
import librosa
import librosa.display
import openl3
import openl3.models
import crema
import matplotlib.pyplot as plt
from tqdm import tqdm

from functools import lru_cache

def torch_device_default():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=32)
def cached_load_audio(audio_path, sr):
    return librosa.load(str(audio_path), sr=sr)


@lru_cache(maxsize=1)
def load_beat_model(device: str):
    return Audio2Beats(checkpoint_path="final0", device=device, dbn=False)


@lru_cache(maxsize=1)
def load_yamnet_model():
    return tf_hub.load('https://tfhub.dev/google/yamnet/1')


@lru_cache(maxsize=1)
def load_openl3_model():
    return openl3.models.load_audio_embedding_model(
        'mel256', 'music', 512, frontend='kapre'
    )


@lru_cache(maxsize=1)
def load_crema_model():
    return crema.models.chord.ChordModel()


def get_track_basename(audio_path):
    track_bn = os.path.basename(audio_path).split('.')[0]
    if track_bn == 'audio':
        track_bn = os.path.dirname(audio_path).split('/')[-1]
    return track_bn


def ensure_beat_boundaries(beats_arr, track_dur):
    if beats_arr[0] > 0:
        beats_arr = np.insert(beats_arr, 0, 0)
    if beats_arr[-1] < track_dur:
        beats_arr = np.append(beats_arr, track_dur)
    return beats_arr


def beats(audio_path, output_dir='beats', recompute=False):
    os.makedirs(output_dir, exist_ok=True)
    track_bn = get_track_basename(audio_path)
    beat_path = os.path.join(output_dir, f'{track_bn}_beats.npz')
    
    if recompute or not os.path.exists(beat_path):
        beat_model = load_beat_model(torch_device_default())
        y, sr = cached_load_audio(audio_path, 22050)
        beats, downbeats = beat_model(y, sr)
        
        track_dur = np.round(librosa.get_duration(path=audio_path), 3)
        beats = ensure_beat_boundaries(beats, track_dur)
        downbeats = ensure_beat_boundaries(downbeats, track_dur)
        
        np.savez(beat_path, beats=beats, downbeats=downbeats)
    return np.load(beat_path)


def beat_sync(feat, track_beats, feat_sr):
    """Synchronizes embeddings to beats, removing invalid beat frames.

    Args:
        feat (np.ndarray): Array of embeddings (D X T).
        track_beats (np.ndarray): Array of beat timestamps (T,).
        feat_sr (float): Sample rate of the features.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Beat-synchronized embeddings (D X B).
            - np.ndarray: Corrected beat timestamps (B,).
    """
    beat_frames = librosa.time_to_samples(track_beats, sr=feat_sr)
    # make the last frame at least as long as the num feature frames.
    # If we don't do this, then librosa.util.sync will pad the beat_frames array,
    # which will cause the synced_feat to have an extra frame.
    if beat_frames[-1] < feat.shape[1]:
        beat_frames[-1] = feat.shape[1]

    # Find bad beat frames: repeated values or out of bounds
    rep_idx = np.where(np.diff(beat_frames) == 0)[0]
    over_idx = np.where(beat_frames >= feat.shape[1])[0]

    # Combine and remove bad indices, but keep the last frame if it's out of bounds
    bad_idx = list(set(rep_idx).union(set(over_idx)) - {len(beat_frames) - 1})

    if bad_idx:
        beat_frames = np.delete(beat_frames, bad_idx)
        track_beats = np.delete(track_beats, bad_idx)

    # Synchronize embeddings to the cleaned beat frames
    synced_feat = librosa.util.sync(feat, beat_frames, aggregate=np.median, pad=True)
    if synced_feat.shape[1] != len(track_beats) - 1:
        raise ValueError(f"Beat synchronization shape mismatch! {synced_feat.shape[1]} != {len(track_beats)} - 1")
    return synced_feat, track_beats


def compute_and_sync_feature(audio_path, output_dir, feature_name, beats_dir, recompute, compute_fn):
    os.makedirs(output_dir, exist_ok=True)
    track_bn = get_track_basename(audio_path)
    feat_path = os.path.join(output_dir, f'{track_bn}_{feature_name}.npz')
    
    if recompute or not os.path.exists(feat_path):
        feat, emb_sr = compute_fn(audio_path)
        track_beats = beats(audio_path, beats_dir, recompute)['beats']
        feat_sync, feat_boundaries = beat_sync(feat, track_beats, emb_sr)
        np.savez(feat_path, feature=feat_sync, ts=feat_boundaries)
    return np.load(feat_path)


def yamnet_emb(audio_path, output_dir='yamnet', beats_dir='beats', recompute=False):
    def compute_yamnet(audio_path):
        yamnet_model = load_yamnet_model()
        audio, sr = cached_load_audio(audio_path, 16000)
        _, yamnet_emb, _ = yamnet_model(audio)
        return yamnet_emb.numpy().T, 1.0/0.48
    
    return compute_and_sync_feature(audio_path, output_dir, 'yamnet', beats_dir, recompute, compute_yamnet)


def openl3_emb(audio_path, output_dir='openl3', beats_dir='beats', recompute=False):
    def compute_openl3(audio_path):
        openl3_model = load_openl3_model()
        y, sr = cached_load_audio(audio_path, 22050)
        emb, ts = openl3.get_audio_embedding(
            y, sr, model=openl3_model, 
            input_repr='mel256', content_type='music', embedding_size=512
        )
        return emb.T, 1.0 / (ts[1] - ts[0])
    
    return compute_and_sync_feature(audio_path, output_dir, 'openl3', beats_dir, recompute, compute_openl3)


def crema_emb(audio_path, output_dir='crema', beats_dir='beats', recompute=False):
    def compute_crema(audio_path):
        with tf.device('CPU'):
            chord_model = load_crema_model()
            crema_out = chord_model.outputs(filename=str(audio_path))
        
        crema_op = chord_model.pump.ops[2]
        emb_sr = crema_op.sr / crema_op.hop_length
        crema_emb = np.concatenate([crema_out['chord_bass'], crema_out['chord_pitch']], axis=1)
        return crema_emb.T, emb_sr
    
    return compute_and_sync_feature(audio_path, output_dir, 'crema', beats_dir, recompute, compute_crema)


def mfcc(audio_path, output_dir='mfcc', beats_dir='beats', recompute=False):
    def compute_mfcc(audio_path):
        y, sr = cached_load_audio(audio_path, 22050)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=40, 
            hop_length=4096, n_fft=8192, lifter=0.6)
        normalized_mfcc = (mfcc - np.mean(mfcc, axis=1)[:, None]) / np.std(mfcc, axis=1, ddof=1)[:,None]
        return normalized_mfcc, sr / 4096.0
    
    return compute_and_sync_feature(audio_path, output_dir, 'mfcc', beats_dir, recompute, compute_mfcc)


def tempogram(audio_path, output_dir='tempogram', beats_dir='beats', recompute=False):
    def compute_tempogram(audio_path):
        y, sr = cached_load_audio(audio_path, 22050)
        novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempogram = librosa.feature.tempogram(
            onset_envelope=novelty, sr=sr, 
            hop_length=512, win_length=384
        )
        return tempogram, sr / 512.0
    
    return compute_and_sync_feature(audio_path, output_dir, 'tempogram', beats_dir, recompute, compute_tempogram)


def plot(out, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    features = out['feature']
    feat_dim = features.shape[0]
    # Use specshow to plot the features. 
    mesh = librosa.display.specshow(
        features, 
        x_axis='time', x_coords=out['ts'], 
        y_axis='none', y_coords = np.arange(feat_dim + 1),
        ax=ax
    )    
    fig.colorbar(mesh, ax=ax)
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Feature Extractor')
    parser.add_argument('audio_dir', help='Path to directory containing audio files')
    parser.add_argument('out_dir', help='Path to feature output directory')

    parser.add_argument(
        '--recompute', 
        action='store_true', 
        help='Optional Flag to ignore existing computed features in the output directory and recompute everything'
    )
    parser.set_defaults(recompute=False)

    kwargs = parser.parse_args()

    # collect all audio files in the audio_dir
    audio_files = [f for f in os.listdir(kwargs.audio_dir) if os.path.isfile(os.path.join(kwargs.audio_dir, f))]
    audio_paths = [os.path.join(kwargs.audio_dir, f) for f in audio_files]
    feats_dir = os.path.join(kwargs.out_dir, 'feats')
    beats_dir = os.path.join(kwargs.out_dir, 'beats')

    for audio_path in tqdm(audio_paths):
        try:
            mfcc(audio_path, feats_dir, beats_dir, kwargs.recompute)
            tempogram(audio_path, feats_dir, beats_dir, kwargs.recompute)
            crema_emb(audio_path, feats_dir, beats_dir, kwargs.recompute)
            openl3_emb(audio_path, feats_dir, beats_dir, kwargs.recompute)
            yamnet_emb(audio_path, feats_dir, beats_dir, kwargs.recompute)
        except Exception as e:
            print(f"Failed to process {audio_path}: {type(e).__name__}: {e}")
    
    