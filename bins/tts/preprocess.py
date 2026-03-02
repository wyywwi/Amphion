# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faulthandler

faulthandler.enable()

import os
import argparse
import json
import pyworld as pw
import numpy as np
from multiprocessing import cpu_count


from utils.util import load_config, str2bool
from preprocessors.processor import preprocess_dataset, prepare_align
from preprocessors.metadata import cal_metadata
from processors import (
    acoustic_extractor,
    content_extractor,
    data_augment,
    phone_extractor,
)


def load_dataset_metadata(dataset, output_path, dataset_types):
    metadata = []
    dataset_output = os.path.join(output_path, dataset)
    for dataset_type in dataset_types:
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        if not os.path.exists(dataset_file):
            return None
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))
    return metadata


def infer_mel_frames(mel, n_mel):
    if mel.ndim != 2:
        return None
    if mel.shape[0] == n_mel:
        return mel.shape[1]
    if mel.shape[1] == n_mel:
        return mel.shape[0]
    return None


def is_acoustic_features_complete(dataset, output_path, cfg, dataset_types):
    metadata = load_dataset_metadata(dataset, output_path, dataset_types)
    if metadata is None or len(metadata) == 0:
        return False, "missing split metadata"

    dataset_output = os.path.join(output_path, dataset)
    n_mel = cfg.preprocess.n_mel
    audio_ext = "wav" if cfg.task_type == "tts" else "npy"

    for utt in metadata:
        uid = utt["Uid"]
        total_frames = None
        num_phones = None

        if cfg.preprocess.extract_duration:
            duration_path = os.path.join(
                dataset_output, cfg.preprocess.duration_dir, uid + ".npy"
            )
            lab_path = os.path.join(dataset_output, cfg.preprocess.lab_dir, uid + ".txt")
            if not os.path.exists(duration_path) or os.path.getsize(duration_path) == 0:
                return False, f"missing duration: {uid}"
            if not os.path.exists(lab_path) or os.path.getsize(lab_path) == 0:
                return False, f"missing phone lab: {uid}"

            durations = np.asarray(np.load(duration_path)).reshape(-1)
            if durations.size == 0:
                return False, f"empty duration: {uid}"
            total_frames = int(np.sum(durations))
            num_phones = int(len(durations))
            if total_frames <= 0:
                return False, f"invalid duration sum: {uid}"

        if cfg.preprocess.extract_mel:
            mel_path = os.path.join(dataset_output, cfg.preprocess.mel_dir, uid + ".npy")
            if not os.path.exists(mel_path) or os.path.getsize(mel_path) == 0:
                return False, f"missing mel: {uid}"
            mel = np.load(mel_path)
            mel_frames = infer_mel_frames(mel, n_mel)
            if mel_frames is None:
                return False, f"invalid mel shape: {uid}, shape={tuple(mel.shape)}"
            if total_frames is not None and mel_frames != total_frames:
                return (
                    False,
                    f"mel-duration mismatch: {uid}, mel={mel_frames}, dur={total_frames}",
                )

        if cfg.preprocess.extract_pitch:
            pitch_path = os.path.join(
                dataset_output, cfg.preprocess.pitch_dir, uid + ".npy"
            )
            if not os.path.exists(pitch_path) or os.path.getsize(pitch_path) == 0:
                return False, f"missing pitch: {uid}"
            pitch = np.asarray(np.load(pitch_path)).reshape(-1)
            if total_frames is not None and len(pitch) != total_frames:
                return (
                    False,
                    f"pitch-duration mismatch: {uid}, pitch={len(pitch)}, dur={total_frames}",
                )
            if cfg.preprocess.extract_duration:
                phone_pitch_path = os.path.join(
                    dataset_output, cfg.preprocess.phone_pitch_dir, uid + ".npy"
                )
                if not os.path.exists(phone_pitch_path) or os.path.getsize(
                    phone_pitch_path
                ) == 0:
                    return False, f"missing phone pitch: {uid}"
                phone_pitch = np.asarray(np.load(phone_pitch_path)).reshape(-1)
                if len(phone_pitch) != num_phones:
                    return (
                        False,
                        f"phone-pitch duration mismatch: {uid}, phone_pitch={len(phone_pitch)}, phones={num_phones}",
                    )

        if cfg.preprocess.extract_energy:
            energy_path = os.path.join(
                dataset_output, cfg.preprocess.energy_dir, uid + ".npy"
            )
            if not os.path.exists(energy_path) or os.path.getsize(energy_path) == 0:
                return False, f"missing energy: {uid}"
            energy = np.asarray(np.load(energy_path)).reshape(-1)
            if total_frames is not None and len(energy) != total_frames:
                return (
                    False,
                    f"energy-duration mismatch: {uid}, energy={len(energy)}, dur={total_frames}",
                )
            if cfg.preprocess.extract_duration:
                phone_energy_path = os.path.join(
                    dataset_output, cfg.preprocess.phone_energy_dir, uid + ".npy"
                )
                if not os.path.exists(phone_energy_path) or os.path.getsize(
                    phone_energy_path
                ) == 0:
                    return False, f"missing phone energy: {uid}"
                phone_energy = np.asarray(np.load(phone_energy_path)).reshape(-1)
                if len(phone_energy) != num_phones:
                    return (
                        False,
                        f"phone-energy duration mismatch: {uid}, phone_energy={len(phone_energy)}, phones={num_phones}",
                    )

        if cfg.preprocess.extract_audio:
            audio_path = os.path.join(
                dataset_output, cfg.preprocess.audio_dir, uid + "." + audio_ext
            )
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return False, f"missing audio: {uid}"

    return True, ""


def is_phone_sequences_complete(dataset, output_path, cfg, dataset_types):
    metadata = load_dataset_metadata(dataset, output_path, dataset_types)
    if metadata is None or len(metadata) == 0:
        return False, "missing split metadata"

    phone_dir = os.path.join(output_path, dataset, cfg.preprocess.phone_dir)
    if not os.path.isdir(phone_dir):
        return False, "phone directory not found"

    for utt in metadata:
        uid = utt["Uid"]
        phone_path = os.path.join(phone_dir, uid + ".phone")
        if not os.path.exists(phone_path) or os.path.getsize(phone_path) == 0:
            return False, f"missing phone sequence: {uid}"

    if cfg.preprocess.phone_extractor != "lexicon":
        symbol_file = os.path.join(
            output_path, dataset, cfg.preprocess.symbols_dict
        )
        if not os.path.exists(symbol_file) or os.path.getsize(symbol_file) == 0:
            return False, "missing phone symbol table"

    return True, ""


def mel_min_max_stats_exist(dataset, output_path, cfg):
    stats_dir = os.path.join(
        output_path, dataset, cfg.preprocess.mel_min_max_stats_dir
    )
    return os.path.exists(os.path.join(stats_dir, "mel_min.npy")) and os.path.exists(
        os.path.join(stats_dir, "mel_max.npy")
    )


def normalization_stat_exists(dataset, output_path, feat_dir):
    stat_path = os.path.join(output_path, dataset, f"{feat_dir}_stat.npy")
    return os.path.exists(stat_path)


def validate_phone_extraction_dependencies(cfg):
    if not cfg.preprocess.extract_phone:
        return
    if len(cfg.dataset) == 0:
        return

    try:
        probe_extractor = phone_extractor.phoneExtractor(
            cfg, dataset_name=cfg.dataset[0]
        )
        probe_extractor.extract_phone("dependency check")
    except LookupError as e:
        raise RuntimeError(
            "Phone extraction dependency check failed. "
            "Please install NLTK resource 'averaged_perceptron_tagger_eng' "
            "before running stage 1."
        ) from e


def extract_acoustic_features(dataset, output_path, cfg, dataset_types, n_workers=1):
    """Extract acoustic features of utterances in the dataset

    Args:
        dataset (str): name of dataset, e.g. opencpop
        output_path (str): directory that stores train, test and feature files of datasets
        cfg (dict): dictionary that stores configurations
        n_workers (int, optional): num of processes to extract features in parallel. Defaults to 1.
    """

    metadata = load_dataset_metadata(dataset, output_path, dataset_types)
    if metadata is None:
        raise FileNotFoundError(f"Failed to load split metadata for {dataset}")
    dataset_output = os.path.join(output_path, dataset)

    if n_workers is not None and n_workers > 1:
        acoustic_extractor.extract_utt_acoustic_features_parallel(
            metadata, dataset_output, cfg, n_workers=n_workers
        )
    else:
        acoustic_extractor.extract_utt_acoustic_features_serial(
            metadata, dataset_output, cfg
        )


def extract_content_features(dataset, output_path, cfg, dataset_types, num_workers=1):
    """Extract content features of utterances in the dataset

    Args:
        dataset (str): name of dataset, e.g. opencpop
        output_path (str): directory that stores train, test and feature files of datasets
        cfg (dict): dictionary that stores configurations
    """

    metadata = load_dataset_metadata(dataset, output_path, dataset_types)
    if metadata is None:
        raise FileNotFoundError(f"Failed to load split metadata for {dataset}")

    content_extractor.extract_utt_content_features_dataloader(
        cfg, metadata, num_workers
    )


def extract_phonme_sequences(dataset, output_path, cfg, dataset_types):
    """Extract phoneme features of utterances in the dataset

    Args:
        dataset (str): name of dataset, e.g. opencpop
        output_path (str): directory that stores train, test and feature files of datasets
        cfg (dict): dictionary that stores configurations

    """

    metadata = load_dataset_metadata(dataset, output_path, dataset_types)
    if metadata is None:
        raise FileNotFoundError(f"Failed to load split metadata for {dataset}")
    phone_extractor.extract_utt_phone_sequence(dataset, cfg, metadata)


def preprocess(cfg, args):
    """Preprocess raw data of single or multiple datasets (in cfg.dataset)

    Args:
        cfg (dict): dictionary that stores configurations
        args (ArgumentParser): specify the configuration file and num_workers
    """
    # Specify the output root path to save the processed data
    output_path = cfg.preprocess.processed_dir
    os.makedirs(output_path, exist_ok=True)

    # Fail fast on missing phone dependencies to avoid wasting hours of preprocessing
    validate_phone_extraction_dependencies(cfg)

    # Split train and test sets
    for dataset in cfg.dataset:
        print("Preprocess {}...".format(dataset))

        if args.prepare_alignment:
            # Prepare alignment with MFA
            print("Prepare alignment {}...".format(dataset))
            prepare_align(
                dataset,
                cfg.dataset_path[dataset],
                cfg.preprocess,
                output_path,
                skip_if_completed=args.skip_existing,
            )

        preprocess_dataset(
            dataset,
            cfg.dataset_path[dataset],
            output_path,
            cfg.preprocess,
            cfg.task_type,
            is_custom_dataset=dataset in cfg.use_custom_dataset,
        )

    # Data augmentation: create new wav files with pitch shift, formant shift, equalizer, time stretch
    try:
        assert isinstance(
            cfg.preprocess.data_augment, list
        ), "Please provide a list of datasets need to be augmented."
        if len(cfg.preprocess.data_augment) > 0:
            new_datasets_list = []
            for dataset in cfg.preprocess.data_augment:
                new_datasets = data_augment.augment_dataset(cfg, dataset)
                new_datasets_list.extend(new_datasets)
            cfg.dataset.extend(new_datasets_list) # augmented datasets are added to the dataset list, separately, not directly mixed with the original dataset
            print("Augmentation datasets: ", cfg.dataset)
    except AssertionError as e:
        print("No Data Augmentation.", e, "Current data_augment config: ", cfg.preprocess.data_augment)
    except Exception as e:
        print("No Data Augmentation. Error: ", e)

    # json files
    dataset_types = list()
    dataset_types.append((cfg.preprocess.train_file).split(".")[0]) # train.json
    dataset_types.append((cfg.preprocess.valid_file).split(".")[0]) # valid.json
    if "test" not in dataset_types:
        dataset_types.append("test")    # sometimes we use test.json as valid file, other times we should add it separately
    # fix: there are no legal variable named 'dataset' here, 
    # and 'eval' does not exist in all the datasets used for tts
    # if "eval" in dataset:
    #     dataset_types = ["test"]    # why?

    # Dump metadata of datasets (singers, train/test durations, etc.)
    # Can run when having multiple datasets, only need they have the same dataset_types list
    cal_metadata(cfg, dataset_types)

    # Prepare the acoustic features
    for dataset in cfg.dataset:
        # Skip augmented datasets which do not need to extract acoustic features
        # We will copy acoustic features from the original dataset later
        if (
            "pitch_shift" in dataset
            or "formant_shift" in dataset
            or "equalizer" in dataset
        ):
            continue
        skip_acoustic = False
        if args.skip_existing:
            complete, reason = is_acoustic_features_complete(
                dataset, output_path, cfg, dataset_types
            )
            if complete:
                skip_acoustic = True
                print(
                    "Acoustic features for {} are complete. Skip extraction.".format(
                        dataset
                    )
                )
            else:
                print(
                    "Acoustic features for {} are incomplete ({}). Re-extracting...".format(
                        dataset, reason
                    )
                )
        if not skip_acoustic:
            print(
                "Extracting acoustic features for {} using {} workers ...".format(
                    dataset, args.num_workers
                )
            )
            extract_acoustic_features(
                dataset, output_path, cfg, dataset_types, args.num_workers
            )
        # Calculate the statistics of acoustic features
        if cfg.preprocess.mel_min_max_norm:
            if args.skip_existing and mel_min_max_stats_exist(dataset, output_path, cfg):
                print("Mel min-max stats for {} exist. Skip.".format(dataset))
            else:
                acoustic_extractor.cal_mel_min_max(dataset, output_path, cfg)

        if cfg.preprocess.extract_pitch:
            acoustic_extractor.cal_pitch_statistics(dataset, output_path, cfg)

        if cfg.preprocess.extract_energy:
            acoustic_extractor.cal_energy_statistics(dataset, output_path, cfg)

        if cfg.preprocess.pitch_norm:
            if args.skip_existing and normalization_stat_exists(
                dataset, output_path, cfg.preprocess.pitch_dir
            ):
                print("Pitch normalization stats for {} exist. Skip.".format(dataset))
            else:
                acoustic_extractor.normalize(dataset, cfg.preprocess.pitch_dir, cfg)

        if cfg.preprocess.energy_norm:
            if args.skip_existing and normalization_stat_exists(
                dataset, output_path, cfg.preprocess.energy_dir
            ):
                print("Energy normalization stats for {} exist. Skip.".format(dataset))
            else:
                acoustic_extractor.normalize(dataset, cfg.preprocess.energy_dir, cfg)

    # Copy acoustic features for augmented datasets by creating soft-links
    for dataset in cfg.dataset:
        if "pitch_shift" in dataset:
            src_dataset = dataset.replace("_pitch_shift", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        elif "formant_shift" in dataset:
            src_dataset = dataset.replace("_formant_shift", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        elif "equalizer" in dataset:
            src_dataset = dataset.replace("_equalizer", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        else:
            continue
        dataset_dir = os.path.join(output_path, dataset)
        metadata = []
        for split in ["train", "test"] if not "eval" in dataset else ["test"]:
            metadata_file_path = os.path.join(src_dataset_dir, "{}.json".format(split))
            with open(metadata_file_path, "r") as f:
                metadata.extend(json.load(f))
        print("Copying acoustic features for {}...".format(dataset))
        acoustic_extractor.copy_acoustic_features(
            metadata, dataset_dir, src_dataset_dir, cfg
        )
        if cfg.preprocess.mel_min_max_norm:
            acoustic_extractor.cal_mel_min_max(dataset, output_path, cfg)

        if cfg.preprocess.extract_pitch:
            acoustic_extractor.cal_pitch_statistics(dataset, output_path, cfg)

    # Prepare the content features
    for dataset in cfg.dataset:
        print("Extracting content features for {}...".format(dataset))
        extract_content_features(
            dataset, output_path, cfg, dataset_types, args.num_workers
        )

    # Prepare the phenome sequences
    if cfg.preprocess.extract_phone:
        for dataset in cfg.dataset:
            if args.skip_existing:
                complete, reason = is_phone_sequences_complete(
                    dataset, output_path, cfg, dataset_types
                )
                if complete:
                    print("Phoneme sequence for {} is complete. Skip.".format(dataset))
                    continue
                print(
                    "Phoneme sequence for {} is incomplete ({}). Re-extracting...".format(
                        dataset, reason
                    )
                )
            print("Extracting phoneme sequence for {}...".format(dataset))
            extract_phonme_sequences(dataset, output_path, cfg, dataset_types)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))
    parser.add_argument("--prepare_alignment", type=str2bool, default=False)
    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=True,
        help="Skip completed preprocessing stages after checks.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    preprocess(cfg, args)


if __name__ == "__main__":
    main()
