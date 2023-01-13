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
This script computes features for TTS model prior to training.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_features.py \
    --feature_config_path=<nemo_root_path>/examples/tts/conf/features/feature_22050.yaml \
    --manifest_path=<data_root_path>/fastpitch_manifest.json \
    --audio_path=<data_root_path>/audio \
    --sup_data_path=<data_root_path>/sup_data \
    --num_workers=1
"""

import argparse
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.features import AudioFeaturizer, compute_energy
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_paths, get_sup_data_file_name
from nemo.collections.tts.torch.tts_data_types import Pitch, Energy, Voiced_mask


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute speaker level pitch statistics.",
    )
    parser.add_argument(
        "--feature_config_path", required=True, type=Path, help="Path to feature config file.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_path", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--sup_data_path", required=True, type=Path, help="Path to base directory with supplementary data.",
    )
    parser.add_argument(
        "--save_pitch", default=True, type=bool, help="Whether to save pitch features.",
    )
    parser.add_argument(
        "--save_voiced", default=True, type=bool, help="Whether to save voiced features.",
    )
    parser.add_argument(
        "--save_energy", default=True, type=bool, help="Whether to save energy features.",
    )
    parser.add_argument(
        "--num_workers", default=-1, type=int, help="Number of threads to use. Default -1 to use all CPUs.",
    )
    args = parser.parse_args()
    return args


def _process_entry(
    entry: dict,
    audio_base_path: Path,
    pitch_base_path: Path,
    voiced_base_path: Path,
    energy_base_path: Path,
    featurizer: AudioFeaturizer
) -> None:
    audio_filepath = Path(entry["audio_filepath"])

    audio_path, audio_path_rel = get_audio_paths(audio_path=audio_filepath, base_path=audio_base_path)

    audio = featurizer.read_audio(audio_path)
    sup_data_file_name = get_sup_data_file_name(audio_path_rel)

    if pitch_base_path or voiced_base_path:
        pitch, voiced, _ = featurizer.compute_pitch(audio)

        if pitch_base_path:
            pitch_path = pitch_base_path / sup_data_file_name
            pitch_tensor = torch.from_numpy(pitch)
            torch.save(pitch_tensor, pitch_path)

        if voiced_base_path:
            voiced_path = voiced_base_path / sup_data_file_name
            voiced_tensor = torch.from_numpy(voiced)
            torch.save(voiced_tensor, voiced_path)

    if energy_base_path:
        energy_path = energy_base_path / sup_data_file_name
        spec = featurizer.compute_mel_spectrogram(audio)
        energy = compute_energy(spec)
        energy_tensor = torch.from_numpy(energy)
        torch.save(energy_tensor, energy_path)

    return


def main():
    args = get_args()
    feature_config_path = args.feature_config_path
    manifest_path = args.manifest_path
    audio_base_path = args.audio_path
    sup_base_path = args.sup_data_path
    save_pitch = args.save_pitch
    save_voiced = args.save_voiced
    save_energy = args.save_energy
    num_workers = args.num_workers

    if not os.path.exists(manifest_path):
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not os.path.exists(audio_base_path):
        raise ValueError(f"Audio directory {audio_base_path} does not exist.")

    os.makedirs(sup_base_path, exist_ok=True)

    if save_pitch:
        pitch_base_path = sup_base_path / Pitch.name
        os.makedirs(pitch_base_path, exist_ok=True)
    else:
        pitch_base_path = None

    if save_voiced:
        voiced_base_path = sup_base_path / Voiced_mask.name
        os.makedirs(voiced_base_path, exist_ok=True)
    else:
        voiced_base_path = None

    if save_energy:
        energy_base_path = sup_base_path / Energy.name
        os.makedirs(energy_base_path, exist_ok=True)
    else:
        energy_base_path = None

    feature_config = OmegaConf.load(feature_config_path)
    featurizer = instantiate(feature_config)

    entries = read_manifest(manifest_path)

    Parallel(n_jobs=num_workers)(
        delayed(_process_entry)(
            entry=entry,
            audio_base_path=audio_base_path,
            pitch_base_path=pitch_base_path,
            voiced_base_path=voiced_base_path,
            energy_base_path=energy_base_path,
            featurizer=featurizer
        )
        for entry in tqdm(entries)
    )


if __name__ == "__main__":
    main()
