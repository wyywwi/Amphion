# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import torchaudio
from utils import audio
import csv
import random

from utils.util import has_existed
from text import _clean_text
import librosa
import soundfile as sf
from scipy.io import wavfile

from pathlib import Path
import numpy as np


def textgird_extract(
    corpus_directory,
    output_directory,
    mfa_path=os.path.join(
        "pretrained", "mfa", "montreal-forced-aligner", "bin", "mfa_align"
    ),
    lexicon=os.path.join("text", "lexicon", "librispeech-lexicon.txt"),
    acoustic_model_path=os.path.join(
        "pretrained",
        "mfa",
        "montreal-forced-aligner",
        "pretrained_models",
        "english.zip",
    ),
    jobs="8",
):
    assert os.path.exists(
        corpus_directory
    ), "Please check the directionary contains *.wav, *.lab"
    assert (
        os.path.exists(mfa_path)
        and os.path.exists(lexicon)
        and os.path.exists(acoustic_model_path)
    ), f"Please download the MFA tools to {mfa_path} firstly"
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print(f"MFA results are save in {output_directory}")
    os.system(
        f".{os.path.sep}{mfa_path} {corpus_directory} {lexicon} {acoustic_model_path} {output_directory} -j {jobs} --clean"
    )


def get_lines(file):
    lines = []
    with open(file, encoding="utf-8") as f:
        for line in tqdm(f):
            lines.append(line.strip())
    return lines


def get_uid2utt(ljspeech_path, dataset, cfg):
    index_count = 0
    total_duration = 0

    uid2utt = []
    for l in tqdm(dataset):
        items = l.split("|")
        uid = items[0]
        text = items[2]

        res = {
            "Dataset": "LJSpeech",
            "index": index_count,
            "Singer": "LJSpeech",
            "Uid": uid,
            "Text": text,
        }

        # Duration in wav files
        audio_file = os.path.join(ljspeech_path, "wavs/{}.wav".format(uid))

        res["Path"] = audio_file

        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.size(-1) / sample_rate
        res["Duration"] = duration

        uid2utt.append(res)

        index_count = index_count + 1
        total_duration += duration

    return uid2utt, total_duration / 3600


def split_dataset(
    lines, test_rate=0.05, valid_rate=0.05, test_size=None, valid_size=None
):
    if test_size == None:
        test_size = int(len(lines) * test_rate)
    if valid_size == None:
        valid_size = int(len(lines) * valid_rate)
    random.shuffle(lines)

    train_set = []
    test_set = []
    valid_set = []

    for line in lines[:test_size]:
        test_set.append(line)
    for line in lines[test_size : test_size + valid_size]:
        valid_set.append(line)
    for line in lines[test_size + valid_size :]:
        train_set.append(line)
    return train_set, test_set, valid_set


max_wav_value = 32768.0


def _load_ljspeech_uids(dataset_path):
    metadata_file = os.path.join(dataset_path, "metadata.csv")
    uids = []
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 0:
                continue
            uid = parts[0].strip()
            if uid:
                uids.append(uid)
    return uids


def is_alignment_complete(dataset, dataset_path, cfg, output_path):
    speaker = "LJSpeech"
    raw_speaker_dir = os.path.join(output_path, dataset, cfg.raw_data, speaker)
    textgrid_speaker_dir = os.path.join(output_path, dataset, "TextGrid", speaker)

    if not os.path.isdir(raw_speaker_dir) or not os.path.isdir(textgrid_speaker_dir):
        return False

    try:
        uids = _load_ljspeech_uids(dataset_path)
    except FileNotFoundError:
        return False

    if len(uids) == 0:
        return False

    for uid in uids:
        wav_path = os.path.join(raw_speaker_dir, f"{uid}.wav")
        lab_path = os.path.join(raw_speaker_dir, f"{uid}.lab")
        textgrid_path = os.path.join(textgrid_speaker_dir, f"{uid}.TextGrid")
        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            return False
        if not os.path.isfile(lab_path) or os.path.getsize(lab_path) == 0:
            return False
        if not os.path.isfile(textgrid_path) or os.path.getsize(textgrid_path) == 0:
            return False
    return True


def prepare_align(dataset, dataset_path, cfg, output_path, skip_if_completed=False):
    if skip_if_completed and is_alignment_complete(dataset, dataset_path, cfg, output_path):
        print(
            f"Alignment assets for {dataset} are complete. Skip re-running MFA alignment."
        )
        return

    in_dir = dataset_path
    out_dir = os.path.join(output_path, dataset, cfg.raw_data)
    sampling_rate = cfg.sample_rate
    cleaners = cfg.text_cleaners # english_cleaners
    speaker = "LJSpeech"
    with open(os.path.join(dataset_path, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            output_wav_path = os.path.join(out_dir, speaker, "{}.wav".format(base_name))
            output_lab_path = os.path.join(out_dir, speaker, "{}.lab".format(base_name))

            if os.path.exists(output_wav_path) and os.path.exists(output_lab_path):
                continue

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value   # normalize

                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )

                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
    # Extract textgird with MFA
    textgird_extract(
        corpus_directory=out_dir,
        output_directory=os.path.join(output_path, dataset, "TextGrid"),
    )


def main(output_path, dataset_path, cfg):
    print("-" * 10)
    print("Dataset splits for {}...\n".format("LJSpeech"))

    dataset = "LJSpeech"

    save_dir = os.path.join(output_path, dataset)
    os.makedirs(save_dir, exist_ok=True)
    ljspeech_path = dataset_path

    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")

    speaker = "LJSpeech"
    speakers = [dataset + "_" + speaker]
    singer_lut = {name: i for i, name in enumerate(sorted(speakers))}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)

    if (
        has_existed(train_output_file)
        and has_existed(test_output_file)
        and has_existed(valid_output_file)
    ):
        return

    meta_file = os.path.join(ljspeech_path, "metadata.csv")
    lines = get_lines(meta_file)

    train_set, test_set, valid_set = split_dataset(lines)

    res, hours = get_uid2utt(ljspeech_path, train_set, cfg)

    # Save train
    os.makedirs(save_dir, exist_ok=True)
    with open(train_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Train_hours= {}".format(hours))

    res, hours = get_uid2utt(ljspeech_path, test_set, cfg)

    # Save test
    os.makedirs(save_dir, exist_ok=True)
    with open(test_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Test_hours= {}".format(hours))

    # Fix: use the generated valid set instead of reusing the test set
    res, hours = get_uid2utt(ljspeech_path, valid_set, cfg)

    # Save valid
    os.makedirs(save_dir, exist_ok=True)
    with open(valid_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Valid_hours= {}".format(hours))
