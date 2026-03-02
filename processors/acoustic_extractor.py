# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import multiprocessing as mp
import time
import torch
import numpy as np

import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from utils.io import save_feature, save_txt, save_torch_audio
from utils.util import has_existed
from utils.dsp import compress, audio_to_label
from utils.data_utils import remove_outlier
from preprocessors.metadata import replace_augment_name
from scipy.interpolate import interp1d
from utils.mel import (
    extract_mel_features,
    extract_linear_features,
)

ZERO = 1e-12
_PARALLEL_DATASET_OUTPUT = None
_PARALLEL_CFG = None
_PARALLEL_LOG_TIMING = False
_TACOTRON_STFT_CACHE = {}
_TIMING_KEYS = ("load_audio", "duration", "mel", "energy", "pitch", "save", "total")


def _should_log_acoustic_timing(cfg):
    return bool(getattr(cfg.preprocess, "log_acoustic_timing", True))


def _new_timing():
    return {k: 0.0 for k in _TIMING_KEYS}


def _new_timing_stats():
    return {k: [] for k in _TIMING_KEYS}


def _collect_timing(stats, timing):
    if stats is None or timing is None:
        return
    for k in _TIMING_KEYS:
        if k in timing:
            stats[k].append(float(timing[k]))


def _print_timing_report(stats, mode, num_utts, wall_time):
    if stats is None or num_utts == 0:
        return
    total_arr = np.asarray(stats["total"], dtype=np.float64)
    if total_arr.size == 0:
        return
    p50_total = float(np.percentile(total_arr, 50))
    p90_total = float(np.percentile(total_arr, 90))
    avg_total = float(np.mean(total_arr))
    print(
        "[AcousticTiming] mode={} utts={} wall={:.2f}s avg_total={:.4f}s p50_total={:.4f}s p90_total={:.4f}s".format(
            mode, num_utts, wall_time, avg_total, p50_total, p90_total
        )
    )

    stage_keys = [k for k in _TIMING_KEYS if k != "total"]
    stage_parts = []
    for key in stage_keys:
        arr = np.asarray(stats[key], dtype=np.float64)
        if arr.size == 0:
            continue
        stage_parts.append(
            "{}:avg={:.4f}s,p50={:.4f}s,p90={:.4f}s".format(
                key,
                float(np.mean(arr)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)),
            )
        )
    if stage_parts:
        print("[AcousticTiming][{}] {}".format(mode, " | ".join(stage_parts)))


def _save_feature_with_timing(
    dataset_output, feature_dir, uid, feature, timing=None, overrides=True
):
    t0 = time.perf_counter() if timing is not None else None
    save_feature(dataset_output, feature_dir, uid, feature, overrides=overrides)
    if timing is not None:
        timing["save"] += time.perf_counter() - t0


def _save_txt_with_timing(
    dataset_output, feature_dir, uid, feature, timing=None, overrides=True
):
    t0 = time.perf_counter() if timing is not None else None
    save_txt(dataset_output, feature_dir, uid, feature, overrides=overrides)
    if timing is not None:
        timing["save"] += time.perf_counter() - t0


def _save_torch_audio_with_timing(
    dataset_output, feature_dir, uid, wav_torch, fs, timing=None, overrides=True
):
    t0 = time.perf_counter() if timing is not None else None
    save_torch_audio(
        dataset_output, feature_dir, uid, wav_torch, fs, overrides=overrides
    )
    if timing is not None:
        timing["save"] += time.perf_counter() - t0


def _get_cached_tacotron_stft(cfg_preprocess, device_key):
    from utils.stft import TacotronSTFT

    cache_key = (
        int(cfg_preprocess.sample_rate),
        int(cfg_preprocess.win_size),
        int(cfg_preprocess.hop_size),
        int(cfg_preprocess.n_fft),
        int(cfg_preprocess.n_mel),
        float(cfg_preprocess.fmin),
        float(cfg_preprocess.fmax),
        str(device_key),
    )
    stft = _TACOTRON_STFT_CACHE.get(cache_key)
    if stft is None:
        stft = TacotronSTFT(
            sampling_rate=cfg_preprocess.sample_rate,
            win_length=cfg_preprocess.win_size,
            hop_length=cfg_preprocess.hop_size,
            filter_length=cfg_preprocess.n_fft,
            n_mel_channels=cfg_preprocess.n_mel,
            mel_fmin=cfg_preprocess.fmin,
            mel_fmax=cfg_preprocess.fmax,
        )
        _TACOTRON_STFT_CACHE[cache_key] = stft
    return stft


def _extract_taco_mel_energy(wav_torch, _stft):
    audio = torch.clip(wav_torch.unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    mel, energy = _stft.mel_spectrogram(audio)
    mel = torch.squeeze(mel, 0)
    energy = torch.squeeze(energy, 0).cpu().numpy().astype(np.float32)
    return mel, energy


def _is_gpu_heavy_acoustic_path(cfg):
    reasons = []
    if bool(getattr(cfg.preprocess, "extract_mel", False)) and getattr(
        cfg.preprocess, "mel_extract_mode", None
    ) == "taco":
        reasons.append("mel_extract_mode=taco")

    if bool(getattr(cfg.preprocess, "extract_energy", False)) and getattr(
        cfg.preprocess, "energy_extract_mode", None
    ) == "from_tacotron_stft":
        reasons.append("energy_extract_mode=from_tacotron_stft")

    if bool(getattr(cfg.preprocess, "extract_acoustic_token", False)):
        reasons.append("extract_acoustic_token=true")

    return reasons


def _is_gpu_batch_mel_energy_enabled(cfg):
    return bool(getattr(cfg.preprocess, "gpu_batch_mel_energy", False))


def _dispatch_extract_utt_acoustic_features(
    dataset_output, cfg, utt, return_timing=False
):
    if cfg.task_type == "tts":
        return extract_utt_acoustic_features_tts(
            dataset_output, cfg, utt, return_timing=return_timing
        )
    if cfg.task_type == "svc":
        return extract_utt_acoustic_features_svc(
            dataset_output, cfg, utt, return_timing=return_timing
        )
    if cfg.task_type == "vocoder":
        return extract_utt_acoustic_features_vocoder(
            dataset_output, cfg, utt, return_timing=return_timing
        )
    if cfg.task_type == "tta":
        return extract_utt_acoustic_features_tta(
            dataset_output, cfg, utt, return_timing=return_timing
        )
    return None


def _init_parallel_acoustic_worker(dataset_output, cfg, log_timing=False):
    global _PARALLEL_DATASET_OUTPUT, _PARALLEL_CFG, _PARALLEL_LOG_TIMING
    _PARALLEL_DATASET_OUTPUT = dataset_output
    _PARALLEL_CFG = cfg
    _PARALLEL_LOG_TIMING = bool(log_timing)


def _parallel_extract_worker(utt):
    if _PARALLEL_DATASET_OUTPUT is None or _PARALLEL_CFG is None:
        raise RuntimeError("Parallel acoustic worker is not initialized correctly.")
    uid = utt.get("Uid", "<unknown>")
    try:
        timing = _dispatch_extract_utt_acoustic_features(
            _PARALLEL_DATASET_OUTPUT,
            _PARALLEL_CFG,
            utt,
            return_timing=_PARALLEL_LOG_TIMING,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract acoustic features for utterance {uid}"
        ) from exc
    return {"uid": uid, "timing": timing}


def extract_utt_acoustic_features_parallel(metadata, dataset_output, cfg, n_workers=1):
    """Extract acoustic features from utterances using muliprocess

    Args:
        metadata (dict): dictionary that stores data in train.json and test.json files
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        n_workers (int, optional): num of processes to extract features in parallel. Defaults to 1.

    Returns:
        list: acoustic features
    """
    num_utts = len(metadata)
    if num_utts == 0:
        return
    if _is_gpu_batch_mel_energy_enabled(cfg):
        print(
            "[AcousticExtractor] preprocess.gpu_batch_mel_energy=true is currently disabled in stage 1 mainline path; using per-utterance extraction."
        )

    try:
        workers = int(n_workers)
    except (TypeError, ValueError):
        workers = 1
    requested_workers = max(1, workers)
    workers = max(1, min(workers, num_utts))

    forced_serial_reasons = _is_gpu_heavy_acoustic_path(cfg)
    if workers > 1 and forced_serial_reasons:
        print(
            "[AcousticExtractor] Requested {} workers but auto-switched to serial (1 worker). reason: {}".format(
                requested_workers, ", ".join(forced_serial_reasons)
            )
        )
        workers = 1

    if workers <= 1:
        extract_utt_acoustic_features_serial(metadata, dataset_output, cfg)
        return

    log_timing = _should_log_acoustic_timing(cfg)
    timing_stats = _new_timing_stats() if log_timing else None
    chunksize = max(1, num_utts // (workers * 8))
    wall_start = time.perf_counter()

    # Use spawn so CUDA-dependent branches (e.g., EnCodec token extraction)
    # can run safely in subprocesses.
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_parallel_acoustic_worker,
        initargs=(dataset_output, cfg, log_timing),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_parallel_extract_worker, metadata, chunksize=chunksize),
            total=num_utts,
        ):
            pass
            if log_timing:
                _collect_timing(timing_stats, result.get("timing"))

    if log_timing:
        _print_timing_report(
            timing_stats,
            mode="parallel(workers={})".format(workers),
            num_utts=num_utts,
            wall_time=time.perf_counter() - wall_start,
        )


def avg_phone_feature(feature, duration, interpolation=False):
    feature = feature[: sum(duration)]
    if interpolation:
        nonzero_ids = np.where(feature != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            feature[nonzero_ids],
            fill_value=(feature[nonzero_ids[0]], feature[nonzero_ids[-1]]),
            bounds_error=False,
        )
        feature = interp_fn(np.arange(0, len(feature)))

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            feature[i] = np.mean(feature[pos : pos + d])
        else:
            feature[i] = 0
        pos += d
    feature = feature[: len(duration)]
    return feature


def extract_utt_acoustic_features_serial(metadata, dataset_output, cfg):
    """Extract acoustic features from utterances (in single process)

    Args:
        metadata (dict): dictionary that stores data in train.json and test.json files
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations

    """
    num_utts = len(metadata)
    if num_utts == 0:
        return
    if _is_gpu_batch_mel_energy_enabled(cfg):
        print(
            "[AcousticExtractor] preprocess.gpu_batch_mel_energy=true is currently disabled in stage 1 mainline path; using per-utterance extraction."
        )

    log_timing = _should_log_acoustic_timing(cfg)
    timing_stats = _new_timing_stats() if log_timing else None
    wall_start = time.perf_counter()
    for utt in tqdm(metadata):
        timing = _dispatch_extract_utt_acoustic_features(
            dataset_output, cfg, utt, return_timing=log_timing
        )
        if log_timing:
            _collect_timing(timing_stats, timing)

    if log_timing:
        _print_timing_report(
            timing_stats,
            mode="serial",
            num_utts=num_utts,
            wall_time=time.perf_counter() - wall_start,
        )


def __extract_utt_acoustic_features(dataset_output, cfg, utt, return_timing=False):
    """Extract acoustic features from utterances (in single process)

    Args:
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        utt (dict): utterance info including dataset, singer, uid:{singer}_{song}_{index},
                    path to utternace, duration, utternace index

    """
    from utils import audio, f0, duration

    timing = _new_timing() if return_timing else None
    utt_start = time.perf_counter() if timing is not None else None
    uid = utt["Uid"]
    wav_path = utt["Path"]
    if os.path.exists(os.path.join(dataset_output, cfg.preprocess.raw_data)):
        wav_path = os.path.join(
            dataset_output, cfg.preprocess.raw_data, utt["Singer"], uid + ".wav"
        )

    with torch.no_grad():
        durations = None

        # Load audio data into tensor with sample rate of the config file
        t0 = time.perf_counter() if timing is not None else None
        wav_torch, _ = audio.load_audio_torch(wav_path, cfg.preprocess.sample_rate)
        if timing is not None:
            timing["load_audio"] += time.perf_counter() - t0
        wav = wav_torch.cpu().numpy()

        # extract features
        if cfg.preprocess.extract_duration:
            t0 = time.perf_counter() if timing is not None else None
            durations, phones, start, end = duration.get_duration(
                utt, wav, cfg.preprocess
            )
            if timing is not None:
                timing["duration"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.duration_dir, uid, durations, timing
            )
            _save_txt_with_timing(dataset_output, cfg.preprocess.lab_dir, uid, phones, timing)
            wav = wav[start:end].astype(np.float32)
            wav_torch = torch.from_numpy(wav).to(wav_torch.device)

        use_taco_mel = (
            cfg.preprocess.extract_mel and cfg.preprocess.mel_extract_mode == "taco"
        )
        use_taco_energy = (
            cfg.preprocess.extract_energy
            and cfg.preprocess.energy_extract_mode == "from_tacotron_stft"
        )
        taco_mel = None
        taco_energy = None
        if use_taco_mel or use_taco_energy:
            _stft = _get_cached_tacotron_stft(cfg.preprocess, wav_torch.device)
            t0 = time.perf_counter() if timing is not None else None
            taco_mel, taco_energy = _extract_taco_mel_energy(wav_torch, _stft)
            if timing is not None:
                elapsed = time.perf_counter() - t0
                if use_taco_mel and use_taco_energy:
                    timing["mel"] += elapsed * 0.5
                    timing["energy"] += elapsed * 0.5
                elif use_taco_mel:
                    timing["mel"] += elapsed
                else:
                    timing["energy"] += elapsed

        if cfg.preprocess.extract_linear_spec:
            t0 = time.perf_counter() if timing is not None else None
            linear = extract_linear_features(wav_torch.unsqueeze(0), cfg.preprocess)
            if timing is not None:
                timing["mel"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.linear_dir, uid, linear.cpu().numpy(), timing
            )

        if cfg.preprocess.extract_mel:
            if use_taco_mel:
                mel = taco_mel
            else:
                t0 = time.perf_counter() if timing is not None else None
                mel = extract_mel_features(wav_torch.unsqueeze(0), cfg.preprocess)
                if timing is not None:
                    timing["mel"] += time.perf_counter() - t0
            if cfg.preprocess.extract_duration:
                mel = mel[:, : sum(durations)]
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.mel_dir, uid, mel.cpu().numpy(), timing
            )

        if cfg.preprocess.extract_energy:
            if (
                cfg.preprocess.energy_extract_mode == "from_mel"
                and cfg.preprocess.extract_mel
            ):
                t0 = time.perf_counter() if timing is not None else None
                energy = (mel.exp() ** 2).sum(0).sqrt().cpu().numpy()
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            elif cfg.preprocess.energy_extract_mode == "from_waveform":
                t0 = time.perf_counter() if timing is not None else None
                energy = audio.energy(wav, cfg.preprocess)
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            elif cfg.preprocess.energy_extract_mode == "from_tacotron_stft":
                energy = taco_energy
            else:
                assert cfg.preprocess.energy_extract_mode in [
                    "from_mel",
                    "from_waveform",
                    "from_tacotron_stft",
                ], f"{cfg.preprocess.energy_extract_mode} not in supported energy_extract_mode [from_mel, from_waveform, from_tacotron_stft]"

            if (
                cfg.preprocess.extract_duration
                and use_taco_mel
                and use_taco_energy
                and cfg.preprocess.extract_mel
            ):
                expected_len = sum(durations)
                assert mel.shape[1] >= expected_len and len(energy) >= expected_len, (
                    f"duration and taco mel/energy mismatch for {uid}: "
                    f"duration_sum={expected_len}, mel_frames={mel.shape[1]}, energy_frames={len(energy)}"
                )

            if cfg.preprocess.extract_duration:
                energy = energy[: sum(durations)]
                phone_energy = avg_phone_feature(energy, durations)
                _save_feature_with_timing(
                    dataset_output,
                    cfg.preprocess.phone_energy_dir,
                    uid,
                    phone_energy,
                    timing,
                )

            _save_feature_with_timing(
                dataset_output, cfg.preprocess.energy_dir, uid, energy, timing
            )

        if cfg.preprocess.extract_pitch:
            t0 = time.perf_counter() if timing is not None else None
            pitch = f0.get_f0(wav, cfg.preprocess)
            if cfg.preprocess.extract_duration:
                pitch = pitch[: sum(durations)]
                phone_pitch = avg_phone_feature(pitch, durations, interpolation=True)
                _save_feature_with_timing(
                    dataset_output,
                    cfg.preprocess.phone_pitch_dir,
                    uid,
                    phone_pitch,
                    timing,
                )
            if timing is not None:
                timing["pitch"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.pitch_dir, uid, pitch, timing
            )

            if cfg.preprocess.extract_uv:
                assert isinstance(pitch, np.ndarray)
                uv = pitch != 0
                _save_feature_with_timing(
                    dataset_output, cfg.preprocess.uv_dir, uid, uv, timing
                )

        if cfg.preprocess.extract_audio:
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.audio_dir, uid, wav, timing
            )

        if cfg.preprocess.extract_label:
            if cfg.preprocess.is_mu_law:
                # compress audio
                wav = compress(wav, cfg.preprocess.bits)
            label = audio_to_label(wav, cfg.preprocess.bits)
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.label_dir, uid, label, timing
            )

        if cfg.preprocess.extract_acoustic_token:
            if cfg.preprocess.acoustic_token_extractor == "Encodec":
                from utils.tokenizer import extract_encodec_token

                codes = extract_encodec_token(wav_path)
                _save_feature_with_timing(
                    dataset_output, cfg.preprocess.acoustic_token_dir, uid, codes, timing
                )

    if timing is not None:
        timing["total"] = time.perf_counter() - utt_start
        return timing
    return None


# TODO: refactor extract_utt_acoustic_features_task function due to many duplicated code
def extract_utt_acoustic_features_tts(
    dataset_output, cfg, utt, return_timing=False
):
    """Extract acoustic features from utterances (in single process)

    Args:
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        utt (dict): utterance info including dataset, singer, uid:{singer}_{song}_{index},
                    path to utternace, duration, utternace index

    """
    from utils import audio, f0, duration

    timing = _new_timing() if return_timing else None
    utt_start = time.perf_counter() if timing is not None else None
    uid = utt["Uid"]
    wav_path = utt["Path"]
    if os.path.exists(os.path.join(dataset_output, cfg.preprocess.raw_data)):
        wav_path = os.path.join(
            dataset_output, cfg.preprocess.raw_data, utt["Singer"], uid + ".wav"
        )
        if not os.path.exists(wav_path):
            wav_path = os.path.join(
                dataset_output, cfg.preprocess.raw_data, utt["Singer"], uid + ".flac"
            )

        assert os.path.exists(wav_path)

    with torch.no_grad():
        durations = None

        # Load audio data into tensor with sample rate of the config file
        t0 = time.perf_counter() if timing is not None else None
        wav_torch, _ = audio.load_audio_torch(wav_path, cfg.preprocess.sample_rate)
        if timing is not None:
            timing["load_audio"] += time.perf_counter() - t0
        wav = wav_torch.cpu().numpy()

        # extract features
        if cfg.preprocess.extract_duration:
            t0 = time.perf_counter() if timing is not None else None
            durations, phones, start, end = duration.get_duration(
                utt, wav, cfg.preprocess
            )
            if timing is not None:
                timing["duration"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.duration_dir, uid, durations, timing
            )
            _save_txt_with_timing(dataset_output, cfg.preprocess.lab_dir, uid, phones, timing)
            wav = wav[start:end].astype(np.float32)
            wav_torch = torch.from_numpy(wav).to(wav_torch.device)

        use_taco_mel = (
            cfg.preprocess.extract_mel and cfg.preprocess.mel_extract_mode == "taco"
        )
        use_taco_energy = (
            cfg.preprocess.extract_energy
            and cfg.preprocess.energy_extract_mode == "from_tacotron_stft"
        )
        taco_mel = None
        taco_energy = None
        if use_taco_mel or use_taco_energy:
            _stft = _get_cached_tacotron_stft(cfg.preprocess, wav_torch.device)
            t0 = time.perf_counter() if timing is not None else None
            taco_mel, taco_energy = _extract_taco_mel_energy(wav_torch, _stft)
            if timing is not None:
                elapsed = time.perf_counter() - t0
                if use_taco_mel and use_taco_energy:
                    timing["mel"] += elapsed * 0.5
                    timing["energy"] += elapsed * 0.5
                elif use_taco_mel:
                    timing["mel"] += elapsed
                else:
                    timing["energy"] += elapsed

        if cfg.preprocess.extract_linear_spec:
            from utils.mel import extract_linear_features

            t0 = time.perf_counter() if timing is not None else None
            linear = extract_linear_features(wav_torch.unsqueeze(0), cfg.preprocess)
            if timing is not None:
                timing["mel"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.linear_dir, uid, linear.cpu().numpy(), timing
            )

        if cfg.preprocess.extract_mel:
            from utils.mel import extract_mel_features

            if use_taco_mel:
                mel = taco_mel
            else:
                t0 = time.perf_counter() if timing is not None else None
                mel = extract_mel_features(wav_torch.unsqueeze(0), cfg.preprocess)
                if timing is not None:
                    timing["mel"] += time.perf_counter() - t0
            if cfg.preprocess.extract_duration:
                mel = mel[:, : sum(durations)]
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.mel_dir, uid, mel.cpu().numpy(), timing
            )

        if cfg.preprocess.extract_energy:
            if (
                cfg.preprocess.energy_extract_mode == "from_mel"
                and cfg.preprocess.extract_mel
            ):
                t0 = time.perf_counter() if timing is not None else None
                energy = (mel.exp() ** 2).sum(0).sqrt().cpu().numpy()
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            elif cfg.preprocess.energy_extract_mode == "from_waveform":
                t0 = time.perf_counter() if timing is not None else None
                energy = audio.energy(wav, cfg.preprocess)
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            elif cfg.preprocess.energy_extract_mode == "from_tacotron_stft":
                energy = taco_energy
            else:
                assert cfg.preprocess.energy_extract_mode in [
                    "from_mel",
                    "from_waveform",
                    "from_tacotron_stft",
                ], f"{cfg.preprocess.energy_extract_mode} not in supported energy_extract_mode [from_mel, from_waveform, from_tacotron_stft]"

            if (
                cfg.preprocess.extract_duration
                and use_taco_mel
                and use_taco_energy
                and cfg.preprocess.extract_mel
            ):
                expected_len = sum(durations)
                assert mel.shape[1] >= expected_len and len(energy) >= expected_len, (
                    f"duration and taco mel/energy mismatch for {uid}: "
                    f"duration_sum={expected_len}, mel_frames={mel.shape[1]}, energy_frames={len(energy)}"
                )
            if cfg.preprocess.extract_duration:
                energy = energy[: sum(durations)]
                phone_energy = avg_phone_feature(energy, durations)
                _save_feature_with_timing(
                    dataset_output,
                    cfg.preprocess.phone_energy_dir,
                    uid,
                    phone_energy,
                    timing,
                )

            _save_feature_with_timing(
                dataset_output, cfg.preprocess.energy_dir, uid, energy, timing
            )

        if cfg.preprocess.extract_pitch:
            t0 = time.perf_counter() if timing is not None else None
            pitch = f0.get_f0(wav, cfg.preprocess)
            if cfg.preprocess.extract_duration:
                pitch = pitch[: sum(durations)]
                phone_pitch = avg_phone_feature(pitch, durations, interpolation=True)
                _save_feature_with_timing(
                    dataset_output,
                    cfg.preprocess.phone_pitch_dir,
                    uid,
                    phone_pitch,
                    timing,
                )
            if timing is not None:
                timing["pitch"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.pitch_dir, uid, pitch, timing
            )

            if cfg.preprocess.extract_uv:
                assert isinstance(pitch, np.ndarray)
                uv = pitch != 0
                _save_feature_with_timing(
                    dataset_output, cfg.preprocess.uv_dir, uid, uv, timing
                )

        if cfg.preprocess.extract_audio:
            _save_torch_audio_with_timing(
                dataset_output,
                cfg.preprocess.audio_dir,
                uid,
                wav_torch,
                cfg.preprocess.sample_rate,
                timing=timing,
            )

        if cfg.preprocess.extract_label:
            if cfg.preprocess.is_mu_law:
                # compress audio
                wav = compress(wav, cfg.preprocess.bits)
            label = audio_to_label(wav, cfg.preprocess.bits)
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.label_dir, uid, label, timing
            )

        if cfg.preprocess.extract_acoustic_token:
            if cfg.preprocess.acoustic_token_extractor == "Encodec":
                from utils.tokenizer import extract_encodec_token

                codes = extract_encodec_token(wav_path)
                _save_feature_with_timing(
                    dataset_output, cfg.preprocess.acoustic_token_dir, uid, codes, timing
                )

    if timing is not None:
        timing["total"] = time.perf_counter() - utt_start
        return timing
    return None


def extract_utt_acoustic_features_svc(dataset_output, cfg, utt, return_timing=False):
    return __extract_utt_acoustic_features(
        dataset_output, cfg, utt, return_timing=return_timing
    )


def extract_utt_acoustic_features_tta(dataset_output, cfg, utt, return_timing=False):
    return __extract_utt_acoustic_features(
        dataset_output, cfg, utt, return_timing=return_timing
    )


def extract_utt_acoustic_features_vocoder(
    dataset_output, cfg, utt, return_timing=False
):
    """Extract acoustic features from utterances (in single process)

    Args:
        dataset_output (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations
        utt (dict): utterance info including dataset, singer, uid:{singer}_{song}_{index},
                    path to utternace, duration, utternace index

    """
    from utils import audio, f0

    timing = _new_timing() if return_timing else None
    utt_start = time.perf_counter() if timing is not None else None
    uid = utt["Uid"]
    wav_path = utt["Path"]

    with torch.no_grad():
        # Load audio data into tensor with sample rate of the config file
        t0 = time.perf_counter() if timing is not None else None
        wav_torch, _ = audio.load_audio_torch(wav_path, cfg.preprocess.sample_rate)
        if timing is not None:
            timing["load_audio"] += time.perf_counter() - t0
        wav = wav_torch.cpu().numpy()

        # extract features
        if cfg.preprocess.extract_mel:
            from utils.mel import extract_mel_features

            t0 = time.perf_counter() if timing is not None else None
            mel = extract_mel_features(wav_torch.unsqueeze(0), cfg.preprocess)
            if timing is not None:
                timing["mel"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.mel_dir, uid, mel.cpu().numpy(), timing
            )

        if cfg.preprocess.extract_energy:
            if (
                cfg.preprocess.energy_extract_mode == "from_mel"
                and cfg.preprocess.extract_mel
            ):
                t0 = time.perf_counter() if timing is not None else None
                energy = (mel.exp() ** 2).sum(0).sqrt().cpu().numpy()
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            elif cfg.preprocess.energy_extract_mode == "from_waveform":
                t0 = time.perf_counter() if timing is not None else None
                energy = audio.energy(wav, cfg.preprocess)
                if timing is not None:
                    timing["energy"] += time.perf_counter() - t0
            else:
                assert cfg.preprocess.energy_extract_mode in [
                    "from_mel",
                    "from_waveform",
                ], f"{cfg.preprocess.energy_extract_mode} not in supported energy_extract_mode [from_mel, from_waveform, from_tacotron_stft]"

            _save_feature_with_timing(
                dataset_output, cfg.preprocess.energy_dir, uid, energy, timing
            )

        if cfg.preprocess.extract_pitch:
            t0 = time.perf_counter() if timing is not None else None
            pitch = f0.get_f0(wav, cfg.preprocess)
            if timing is not None:
                timing["pitch"] += time.perf_counter() - t0
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.pitch_dir, uid, pitch, timing
            )

            if cfg.preprocess.extract_uv:
                assert isinstance(pitch, np.ndarray)
                uv = pitch != 0
                _save_feature_with_timing(
                    dataset_output, cfg.preprocess.uv_dir, uid, uv, timing
                )

        if cfg.preprocess.extract_amplitude_phase:
            from utils.mel import amplitude_phase_spectrum

            log_amplitude, phase, real, imaginary = amplitude_phase_spectrum(
                wav_torch.unsqueeze(0), cfg.preprocess
            )
            _save_feature_with_timing(
                dataset_output,
                cfg.preprocess.log_amplitude_dir,
                uid,
                log_amplitude,
                timing,
            )
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.phase_dir, uid, phase, timing
            )
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.real_dir, uid, real, timing
            )
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.imaginary_dir, uid, imaginary, timing
            )

        if cfg.preprocess.extract_audio:
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.audio_dir, uid, wav, timing
            )

        if cfg.preprocess.extract_label:
            if cfg.preprocess.is_mu_law:
                # compress audio
                wav = compress(wav, cfg.preprocess.bits)
            label = audio_to_label(wav, cfg.preprocess.bits)
            _save_feature_with_timing(
                dataset_output, cfg.preprocess.label_dir, uid, label, timing
            )

    if timing is not None:
        timing["total"] = time.perf_counter() - utt_start
        return timing
    return None


def cal_normalized_mel(mel, dataset_name, cfg):
    """
    mel: (n_mels, T)
    """
    # mel_min, mel_max: (n_mels)
    mel_min, mel_max = load_mel_extrema(cfg, dataset_name)
    mel_norm = normalize_mel_channel(mel, mel_min, mel_max)
    return mel_norm


def cal_mel_min_max(dataset, output_path, cfg, metadata=None):
    dataset_output = os.path.join(output_path, dataset)

    if metadata is None:
        metadata = []
        for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
            dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
            with open(dataset_file, "r") as f:
                metadata.extend(json.load(f))

    tmp_mel_min = []
    tmp_mel_max = []
    for item in metadata:
        mel_path = os.path.join(
            dataset_output, cfg.preprocess.mel_dir, item["Uid"] + ".npy"
        )
        if not os.path.exists(mel_path):
            continue
        mel = np.load(mel_path)
        if mel.shape[0] != cfg.preprocess.n_mel:
            mel = mel.T
        # mel: (n_mels, T)
        assert mel.shape[0] == cfg.preprocess.n_mel

        tmp_mel_min.append(np.min(mel, axis=-1))
        tmp_mel_max.append(np.max(mel, axis=-1))

    mel_min = np.min(tmp_mel_min, axis=0)
    mel_max = np.max(tmp_mel_max, axis=0)

    ## save mel min max data
    mel_min_max_dir = os.path.join(dataset_output, cfg.preprocess.mel_min_max_stats_dir)
    os.makedirs(mel_min_max_dir, exist_ok=True)

    mel_min_path = os.path.join(mel_min_max_dir, "mel_min.npy")
    mel_max_path = os.path.join(mel_min_max_dir, "mel_max.npy")
    np.save(mel_min_path, mel_min)
    np.save(mel_max_path, mel_max)


def denorm_for_pred_mels(cfg, dataset_name, split, pred):
    """
    Args:
        pred: a list whose every element is (frame_len, n_mels)
    Return:
        similar like pred
    """
    mel_min, mel_max = load_mel_extrema(cfg.preprocess, dataset_name)
    recovered_mels = [
        denormalize_mel_channel(mel.T, mel_min, mel_max).T for mel in pred
    ]

    return recovered_mels


def load_mel_extrema(cfg, dataset_name):
    data_dir = os.path.join(cfg.processed_dir, dataset_name, cfg.mel_min_max_stats_dir)

    min_file = os.path.join(data_dir, "mel_min.npy")
    max_file = os.path.join(data_dir, "mel_max.npy")

    mel_min = np.load(min_file)
    mel_max = np.load(max_file)

    return mel_min, mel_max


def denormalize_mel_channel(mel, mel_min, mel_max):
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    return (mel + 1) / 2 * (mel_max - mel_min + ZERO) + mel_min


def normalize_mel_channel(mel, mel_min, mel_max):
    """
    mel: (n_mels, T)
    mel_min, mel_max: (n_mels)
    """
    mel_min = np.expand_dims(mel_min, -1)
    mel_max = np.expand_dims(mel_max, -1)
    return (mel - mel_min) / (mel_max - mel_min + ZERO) * 2 - 1


def normalize(dataset, feat_dir, cfg):
    dataset_output = os.path.join(cfg.preprocess.processed_dir, dataset)
    print(f"normalize {feat_dir}")

    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max

    scaler = StandardScaler()
    feat_files = os.listdir(os.path.join(dataset_output, feat_dir))

    for feat_file in tqdm(feat_files):
        feat_file = os.path.join(dataset_output, feat_dir, feat_file)
        if not feat_file.endswith(".npy"):
            continue
        feat = np.load(feat_file)
        max_value = max(max_value, max(feat))
        min_value = min(min_value, min(feat))
        scaler.partial_fit(feat.reshape((-1, 1)))
    mean = scaler.mean_[0]
    std = scaler.scale_[0]
    stat = np.array([min_value, max_value, mean, std])
    stat_npy = os.path.join(dataset_output, f"{feat_dir}_stat.npy")
    np.save(stat_npy, stat)
    return mean, std, min_value, max_value


def load_normalized(feat_dir, dataset_name, cfg):
    dataset_output = os.path.join(cfg.preprocess.processed_dir, dataset_name)
    stat_npy = os.path.join(dataset_output, f"{feat_dir}_stat.npy")
    min_value, max_value, mean, std = np.load(stat_npy)
    return mean, std, min_value, max_value


def cal_pitch_statistics_svc(dataset, output_path, cfg, metadata=None):
    # path of dataset
    dataset_dir = os.path.join(output_path, dataset)
    save_dir = os.path.join(dataset_dir, cfg.preprocess.pitch_dir)
    os.makedirs(save_dir, exist_ok=True)
    if has_existed(os.path.join(save_dir, "statistics.json")):
        return

    if metadata is None:
        # load singers and ids
        singers = json.load(open(os.path.join(dataset_dir, "singers.json"), "r"))

        # combine train and test metadata
        metadata = []
        for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
            dataset_file = os.path.join(dataset_dir, "{}.json".format(dataset_type))
            with open(dataset_file, "r") as f:
                metadata.extend(json.load(f))
    else:
        singers = list(set([item["Singer"] for item in metadata]))
        singers = {
            "{}_{}".format(dataset, name): idx for idx, name in enumerate(singers)
        }

    # use different scalers for each singer
    pitch_scalers = [[] for _ in range(len(singers))]
    total_pitch_scalers = [[] for _ in range(len(singers))]

    for utt_info in tqdm(metadata, desc="Loading F0..."):
        # utt = f'{utt_info["Dataset"]}_{utt_info["Uid"]}'
        singer = utt_info["Singer"]
        pitch_path = os.path.join(
            dataset_dir, cfg.preprocess.pitch_dir, utt_info["Uid"] + ".npy"
        )
        # total_pitch contains all pitch including unvoiced frames
        if not os.path.exists(pitch_path):
            continue
        total_pitch = np.load(pitch_path)
        assert len(total_pitch) > 0
        # pitch contains only voiced frames
        pitch = total_pitch[total_pitch != 0]
        spkid = singers[f"{replace_augment_name(dataset)}_{singer}"]

        # update pitch scalers
        pitch_scalers[spkid].extend(pitch.tolist())
        # update total pitch scalers
        total_pitch_scalers[spkid].extend(total_pitch.tolist())

    # save pitch statistics for each singer in dict
    sta_dict = {}
    for singer in tqdm(singers, desc="Singers statistics"):
        spkid = singers[singer]
        # voiced pitch statistics
        mean, std, min, max, median = (
            np.mean(pitch_scalers[spkid]),
            np.std(pitch_scalers[spkid]),
            np.min(pitch_scalers[spkid]),
            np.max(pitch_scalers[spkid]),
            np.median(pitch_scalers[spkid]),
        )

        # total pitch statistics
        mean_t, std_t, min_t, max_t, median_t = (
            np.mean(total_pitch_scalers[spkid]),
            np.std(total_pitch_scalers[spkid]),
            np.min(total_pitch_scalers[spkid]),
            np.max(total_pitch_scalers[spkid]),
            np.median(total_pitch_scalers[spkid]),
        )
        sta_dict[singer] = {
            "voiced_positions": {
                "mean": mean,
                "std": std,
                "median": median,
                "min": min,
                "max": max,
            },
            "total_positions": {
                "mean": mean_t,
                "std": std_t,
                "median": median_t,
                "min": min_t,
                "max": max_t,
            },
        }

    # save statistics
    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        json.dump(sta_dict, f, indent=4, ensure_ascii=False)


def cal_pitch_statistics(dataset, output_path, cfg):
    # path of dataset
    dataset_dir = os.path.join(output_path, dataset)
    if cfg.preprocess.use_phone_pitch:
        pitch_dir = cfg.preprocess.phone_pitch_dir
    else:
        pitch_dir = cfg.preprocess.pitch_dir
    save_dir = os.path.join(dataset_dir, pitch_dir)

    os.makedirs(save_dir, exist_ok=True)
    if has_existed(os.path.join(save_dir, "statistics.json")):
        return
    # load singers and ids
    singers = json.load(open(os.path.join(dataset_dir, "singers.json"), "r"))

    # combine train and test metadata
    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_dir, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    # use different scalers for each singer
    pitch_scalers = [[] for _ in range(len(singers))]
    total_pitch_scalers = [[] for _ in range(len(singers))]

    for utt_info in metadata:
        utt = f'{utt_info["Dataset"]}_{utt_info["Uid"]}'
        singer = utt_info["Singer"]
        pitch_path = os.path.join(dataset_dir, pitch_dir, utt_info["Uid"] + ".npy")
        # total_pitch contains all pitch including unvoiced frames
        if not os.path.exists(pitch_path):
            continue
        total_pitch = np.load(pitch_path)
        assert len(total_pitch) > 0
        # pitch contains only voiced frames
        # pitch = total_pitch[total_pitch != 0]
        if cfg.preprocess.pitch_remove_outlier:
            pitch = remove_outlier(total_pitch)
        spkid = singers[f"{replace_augment_name(dataset)}_{singer}"]

        # update pitch scalers
        pitch_scalers[spkid].extend(pitch.tolist())
        # update total pitch scalers
        total_pitch_scalers[spkid].extend(total_pitch.tolist())

    # save pitch statistics for each singer in dict
    sta_dict = {}
    for singer in singers:
        spkid = singers[singer]
        # voiced pitch statistics
        mean, std, min, max, median = (
            np.mean(pitch_scalers[spkid]),
            np.std(pitch_scalers[spkid]),
            np.min(pitch_scalers[spkid]),
            np.max(pitch_scalers[spkid]),
            np.median(pitch_scalers[spkid]),
        )

        # total pitch statistics
        mean_t, std_t, min_t, max_t, median_t = (
            np.mean(total_pitch_scalers[spkid]),
            np.std(total_pitch_scalers[spkid]),
            np.min(total_pitch_scalers[spkid]),
            np.max(total_pitch_scalers[spkid]),
            np.median(total_pitch_scalers[spkid]),
        )
        sta_dict[singer] = {
            "voiced_positions": {
                "mean": mean,
                "std": std,
                "median": median,
                "min": min,
                "max": max,
            },
            "total_positions": {
                "mean": mean_t,
                "std": std_t,
                "median": median_t,
                "min": min_t,
                "max": max_t,
            },
        }

    # save statistics
    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        json.dump(sta_dict, f, indent=4, ensure_ascii=False)


def cal_energy_statistics(dataset, output_path, cfg):
    # path of dataset
    dataset_dir = os.path.join(output_path, dataset)
    if cfg.preprocess.use_phone_energy:
        energy_dir = cfg.preprocess.phone_energy_dir
    else:
        energy_dir = cfg.preprocess.energy_dir
    save_dir = os.path.join(dataset_dir, energy_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(os.path.join(save_dir, "statistics.json"))
    if has_existed(os.path.join(save_dir, "statistics.json")):
        return
    # load singers and ids
    singers = json.load(open(os.path.join(dataset_dir, "singers.json"), "r"))

    # combine train and test metadata
    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_dir, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    # use different scalers for each singer
    energy_scalers = [[] for _ in range(len(singers))]
    total_energy_scalers = [[] for _ in range(len(singers))]

    for utt_info in metadata:
        utt = f'{utt_info["Dataset"]}_{utt_info["Uid"]}'
        singer = utt_info["Singer"]
        energy_path = os.path.join(dataset_dir, energy_dir, utt_info["Uid"] + ".npy")
        # total_energy contains all energy including unvoiced frames
        if not os.path.exists(energy_path):
            continue
        total_energy = np.load(energy_path)
        assert len(total_energy) > 0
        # energy contains only voiced frames
        # energy = total_energy[total_energy != 0]
        if cfg.preprocess.energy_remove_outlier:
            energy = remove_outlier(total_energy)
        spkid = singers[f"{replace_augment_name(dataset)}_{singer}"]

        # update energy scalers
        energy_scalers[spkid].extend(energy.tolist())
        # update total energyscalers
        total_energy_scalers[spkid].extend(total_energy.tolist())

    # save energy statistics for each singer in dict
    sta_dict = {}
    for singer in singers:
        spkid = singers[singer]
        # voiced energy statistics
        mean, std, min, max, median = (
            np.mean(energy_scalers[spkid]),
            np.std(energy_scalers[spkid]),
            np.min(energy_scalers[spkid]),
            np.max(energy_scalers[spkid]),
            np.median(energy_scalers[spkid]),
        )

        # total energy statistics
        mean_t, std_t, min_t, max_t, median_t = (
            np.mean(total_energy_scalers[spkid]),
            np.std(total_energy_scalers[spkid]),
            np.min(total_energy_scalers[spkid]),
            np.max(total_energy_scalers[spkid]),
            np.median(total_energy_scalers[spkid]),
        )
        sta_dict[singer] = {
            "voiced_positions": {
                "mean": mean,
                "std": std,
                "median": median,
                "min": min,
                "max": max,
            },
            "total_positions": {
                "mean": mean_t,
                "std": std_t,
                "median": median_t,
                "min": min_t,
                "max": max_t,
            },
        }

    # save statistics
    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        json.dump(sta_dict, f, indent=4, ensure_ascii=False)


def copy_acoustic_features(metadata, dataset_dir, src_dataset_dir, cfg):
    """Copy acoustic features from src_dataset_dir to dataset_dir

    Args:
        metadata (dict): dictionary that stores data in train.json and test.json files
        dataset_dir (str): directory to store acoustic features
        src_dataset_dir (str): directory to store acoustic features
        cfg (dict): dictionary that stores configurations

    """

    if cfg.preprocess.extract_mel:
        if not has_existed(os.path.join(dataset_dir, cfg.preprocess.mel_dir)):
            os.makedirs(
                os.path.join(dataset_dir, cfg.preprocess.mel_dir), exist_ok=True
            )
            print(
                "Copying mel features from {} to {}...".format(
                    src_dataset_dir, dataset_dir
                )
            )
            for utt_info in tqdm(metadata):
                src_mel_path = os.path.join(
                    src_dataset_dir, cfg.preprocess.mel_dir, utt_info["Uid"] + ".npy"
                )
                dst_mel_path = os.path.join(
                    dataset_dir, cfg.preprocess.mel_dir, utt_info["Uid"] + ".npy"
                )
                # create soft-links
                if not os.path.exists(dst_mel_path):
                    os.symlink(src_mel_path, dst_mel_path)
    if cfg.preprocess.extract_energy:
        if not has_existed(os.path.join(dataset_dir, cfg.preprocess.energy_dir)):
            os.makedirs(
                os.path.join(dataset_dir, cfg.preprocess.energy_dir), exist_ok=True
            )
            print(
                "Copying energy features from {} to {}...".format(
                    src_dataset_dir, dataset_dir
                )
            )
            for utt_info in tqdm(metadata):
                src_energy_path = os.path.join(
                    src_dataset_dir, cfg.preprocess.energy_dir, utt_info["Uid"] + ".npy"
                )
                dst_energy_path = os.path.join(
                    dataset_dir, cfg.preprocess.energy_dir, utt_info["Uid"] + ".npy"
                )
                # create soft-links
                if not os.path.exists(dst_energy_path):
                    os.symlink(src_energy_path, dst_energy_path)
    if cfg.preprocess.extract_pitch:
        if not has_existed(os.path.join(dataset_dir, cfg.preprocess.pitch_dir)):
            os.makedirs(
                os.path.join(dataset_dir, cfg.preprocess.pitch_dir), exist_ok=True
            )
            print(
                "Copying pitch features from {} to {}...".format(
                    src_dataset_dir, dataset_dir
                )
            )
            for utt_info in tqdm(metadata):
                src_pitch_path = os.path.join(
                    src_dataset_dir, cfg.preprocess.pitch_dir, utt_info["Uid"] + ".npy"
                )
                dst_pitch_path = os.path.join(
                    dataset_dir, cfg.preprocess.pitch_dir, utt_info["Uid"] + ".npy"
                )
                # create soft-links
                if not os.path.exists(dst_pitch_path):
                    os.symlink(src_pitch_path, dst_pitch_path)
        if cfg.preprocess.extract_uv:
            if not has_existed(os.path.join(dataset_dir, cfg.preprocess.uv_dir)):
                os.makedirs(
                    os.path.join(dataset_dir, cfg.preprocess.uv_dir), exist_ok=True
                )
                print(
                    "Copying uv features from {} to {}...".format(
                        src_dataset_dir, dataset_dir
                    )
                )
                for utt_info in tqdm(metadata):
                    src_uv_path = os.path.join(
                        src_dataset_dir, cfg.preprocess.uv_dir, utt_info["Uid"] + ".npy"
                    )
                    dst_uv_path = os.path.join(
                        dataset_dir, cfg.preprocess.uv_dir, utt_info["Uid"] + ".npy"
                    )
                    # create soft-links
                    if not os.path.exists(dst_uv_path):
                        os.symlink(src_uv_path, dst_uv_path)
    if cfg.preprocess.extract_audio:
        if not has_existed(os.path.join(dataset_dir, cfg.preprocess.audio_dir)):
            os.makedirs(
                os.path.join(dataset_dir, cfg.preprocess.audio_dir), exist_ok=True
            )
            print(
                "Copying audio features from {} to {}...".format(
                    src_dataset_dir, dataset_dir
                )
            )
            for utt_info in tqdm(metadata):
                if cfg.task_type == "tts":
                    src_audio_path = os.path.join(
                        src_dataset_dir,
                        cfg.preprocess.audio_dir,
                        utt_info["Uid"] + ".wav",
                    )
                else:
                    src_audio_path = os.path.join(
                        src_dataset_dir,
                        cfg.preprocess.audio_dir,
                        utt_info["Uid"] + ".npy",
                    )
                if cfg.task_type == "tts":
                    dst_audio_path = os.path.join(
                        dataset_dir, cfg.preprocess.audio_dir, utt_info["Uid"] + ".wav"
                    )
                else:
                    dst_audio_path = os.path.join(
                        dataset_dir, cfg.preprocess.audio_dir, utt_info["Uid"] + ".npy"
                    )
                # create soft-links
                if not os.path.exists(dst_audio_path):
                    os.symlink(src_audio_path, dst_audio_path)
    if cfg.preprocess.extract_label:
        if not has_existed(os.path.join(dataset_dir, cfg.preprocess.label_dir)):
            os.makedirs(
                os.path.join(dataset_dir, cfg.preprocess.label_dir), exist_ok=True
            )
            print(
                "Copying label features from {} to {}...".format(
                    src_dataset_dir, dataset_dir
                )
            )
            for utt_info in tqdm(metadata):
                src_label_path = os.path.join(
                    src_dataset_dir, cfg.preprocess.label_dir, utt_info["Uid"] + ".npy"
                )
                dst_label_path = os.path.join(
                    dataset_dir, cfg.preprocess.label_dir, utt_info["Uid"] + ".npy"
                )
                # create soft-links
                if not os.path.exists(dst_label_path):
                    os.symlink(src_label_path, dst_label_path)


def align_duration_mel(dataset, output_path, cfg):
    print("align the duration and mel")

    dataset_dir = os.path.join(output_path, dataset)
    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_dir, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    utt2dur = {}
    for index in tqdm(range(len(metadata))):
        utt_info = metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        mel_path = os.path.join(dataset_dir, cfg.preprocess.mel_dir, uid + ".npy")
        mel = np.load(mel_path).transpose(1, 0)
        duration_path = os.path.join(
            dataset_dir, cfg.preprocess.duration_dir, uid + ".npy"
        )
        duration = np.load(duration_path)
        if sum(duration) != mel.shape[0]:
            duration_sum = sum(duration)
            mel_len = mel.shape[0]
            mismatch = abs(duration_sum - mel_len)
            assert mismatch <= 5, "duration and mel length mismatch!"
            cloned = np.array(duration, copy=True)
            if duration_sum > mel_len:
                for j in range(1, len(duration) - 1):
                    if mismatch == 0:
                        break
                    dur_val = cloned[-j]
                    if dur_val >= mismatch:
                        cloned[-j] -= mismatch
                        mismatch -= dur_val
                        break
                    else:
                        cloned[-j] = 0
                        mismatch -= dur_val

            elif duration_sum < mel_len:
                cloned[-1] += mismatch
            duration = cloned
        utt2dur[utt] = duration
        np.save(duration_path, duration)

    return utt2dur
