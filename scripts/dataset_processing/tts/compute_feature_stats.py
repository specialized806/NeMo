# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is to compute global and speaker-level feature statistics for a given TTS training manifest.

This script should be run after compute_features.py as it uses the precomputed supplemental features.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_feature_stats.py \
    --manifest_path=<data_root_path>/fastpitch_manifest.json \
    --sup_data_path=<data_root_path>/sup_data \
    --pitch_stats_path=<data_root_path>/pitch_stats.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_paths, get_sup_data_file_name
from nemo.collections.tts.torch.tts_data_types import (
    Energy,
    Pitch,
    Voiced_mask
)
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute speaker level pitch statistics.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_dir", required=True, type=Path, help="Path to base directory with audio files.",
    )
    parser.add_argument(
        "--sup_data_path", default=Path("sup_data"), type=Path, help="Path to base directory with supplementary data.",
    )
    parser.add_argument(
        "--stats_path",
        default=Path("feature_stats.json"),
        type=Path,
        help="Path to output JSON file with dataset feature statistics.",
    )
    parser.add_argument(
        "--calculate_pitch", default=True, type=bool, help="Whether to calculate pitch statistics.",
    )
    parser.add_argument(
        "--calculate_energy", default=True, type=bool, help="Whether to calculate energy statistics.",
    )
    args = parser.parse_args()
    return args


def _compute_stats(values: List[torch.Tensor]) -> Tuple[float, float]:
    values_tensor = torch.cat(values, dim=0)
    mean = values_tensor.mean().item()
    std = values_tensor.std(dim=0).item()
    return mean, std


def _compute_stats_for_sup_data_type(
    stat_dict: dict,
    entries: List[dict],
    audio_dir: Path,
    sup_base_path: Path,
    sup_data_type: str,
    voiced_base_path: Path
):
    type_base_path = sup_base_path / sup_data_type
    if not os.path.exists(type_base_path):
        raise ValueError(
            f"Directory {type_base_path} does not exist. Make sure 'sup_data_path' is correct "
            f"and that you have computed the feature using compute_features.py"
        )

    speaker_values = defaultdict(list)
    for entry in tqdm(entries):
        audio_filepath = Path(entry["audio_filepath"])
        _, audio_path_rel = get_audio_paths(audio_path=audio_filepath, base_path=audio_dir)
        sup_data_file_name = get_sup_data_file_name(audio_path_rel)

        voiced_path = voiced_base_path / sup_data_file_name
        if not os.path.exists(voiced_path):
            logging.warning(f"Unable to find voiced file {voiced_path}")
            continue

        type_path = type_base_path / sup_data_file_name
        if not os.path.exists(type_path):
            logging.warning(f"Unable to find feature file {type_path}")
            continue

        data_value = torch.load(type_path)
        voiced_mask = torch.load(voiced_path)
        data_value = data_value[voiced_mask]
        speaker_values["default"].append(data_value)
        if "speaker" in entry:
            speaker_id = entry["speaker"]
            speaker_values[speaker_id].append(data_value)

    mean_key = f"{sup_data_type.lower()}_mean"
    std_key = f"{sup_data_type.lower()}_std"
    for speaker_id, values in speaker_values.items():
        speaker_mean, speaker_std = _compute_stats(values)
        stat_dict[speaker_id][mean_key] = speaker_mean
        stat_dict[speaker_id][std_key] = speaker_std


def main():
    args = get_args()

    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    sup_base_path = args.sup_data_path
    stats_path = args.stats_path
    calculate_pitch = args.calculate_pitch
    calculate_energy = args.calculate_energy

    voiced_base_path = sup_base_path / Voiced_mask.name

    if not os.path.exists(voiced_base_path):
        raise ValueError(
            f"Voiced directory {voiced_base_path} does not exist. Make sure 'sup_data_path' is correct "
            f"and that you have computed features using compute_feature_stats.py"
        )

    entries = read_manifest(manifest_path)

    stat_dict = defaultdict(dict)
    if calculate_pitch:
        _compute_stats_for_sup_data_type(
            stat_dict=stat_dict,
            entries=entries,
            audio_dir=audio_dir,
            sup_base_path=sup_base_path,
            sup_data_type=Pitch.name,
            voiced_base_path=voiced_base_path
        )

    if calculate_energy:
        _compute_stats_for_sup_data_type(
            stat_dict=stat_dict,
            entries=entries,
            audio_dir=audio_dir,
            sup_base_path=sup_base_path,
            sup_data_type=Energy.name,
            voiced_base_path=voiced_base_path
        )

    with open(stats_path, 'w', encoding="utf-8") as stats_f:
        json.dump(stat_dict, stats_f, indent=4)


if __name__ == "__main__":
    main()
