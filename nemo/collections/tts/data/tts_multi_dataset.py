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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch.utils.data
from typing import List, Optional, Tuple

from torch import Tensor

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_paths, get_sup_data_file_name
from nemo.core.classes import Dataset
from nemo.utils import logging

from nemo.collections.tts.torch.tts_data_types import (
    Energy,
    Pitch,
    Voiced_mask
)

SUP_DATA_TYPES = [
    Pitch.name,
    Energy.name,
    Voiced_mask.name
]


@dataclass
class DatasetMeta:
    dataset_name: str
    manifest_path: Path
    audio_dir: Path
    sup_data_dir: Path
    sample_weight: float = 1.0


@dataclass
class DatasetSample:
    audio_dir: Path
    audio_rel_path: Path
    sup_data_dir: Path
    text: str
    speaker: str


@dataclass
class TTSTrainingExample:
    audio: Tensor
    text: str
    speaker_index: int = 0
    pitch: Tensor = None
    energy: Tensor = None
    voiced: Tensor = None


class FeatureProcessor(ABC):

    @abstractmethod
    def process(self, training_example: TTSTrainingExample, field: str) -> None:
        raise NotImplementedError


class TTSMultiDataset(Dataset):

    def __init__(
        self,
        dataset_meta: List[DatasetMeta],
        speaker_path: Optional[Path],
        data_processors: Optional[List[Tuple[str, List[FeatureProcessor]]]] = None,
        sample_rate: int = 22050,
        min_duration: float = 0.0,
        max_duration: float = 0.0
    ):
        super().__init__()

        self.sample_rate = sample_rate

        if speaker_path:
            self.speaker_index_map = self._parse_speaker_file(speaker_path)
        else:
            self.speaker_index_map = None

        self.data_processors = data_processors

        self.sup_data_types = []
        if self.data_processors is not None:
            for data_type, processor in self.data_processors:
                assert data_type in SUP_DATA_TYPES
                self.sup_data_types.append(data_type)

        self.data_samples = []
        self.sample_weights = []
        for dataset in dataset_meta:
            samples, weights = self._process_dataset(dataset, min_duration, max_duration)
            self.data_samples += samples
            self.sample_weights += weights

    @staticmethod
    def _parse_speaker_file(speaker_path):
        # TODO: Parse speaker file with speaker string to index mapping
        return {}

    @staticmethod
    def _filter_by_duration(entries: List[dict], min_duration: float, max_duration: float):
        filtered_entries = []
        total_duration = 0.0
        filtered_duration = 0.0
        for entry in entries:
            duration = entry["duration"]
            total_duration += duration
            if (min_duration and duration < min_duration) or (max_duration and duration > max_duration):
                continue

            filtered_duration += duration
            filtered_entries.append(entry)

        total_hours = total_duration / 3600.0
        filtered_hours = filtered_duration / 3600.0

        logging.info(f"Original # of files: {len(entries)}")
        logging.info(f"Filtered # of files: {len(filtered_entries)}")
        logging.info(f"Original duration: {total_hours} hours")
        logging.info(f"Filtered duration: {filtered_hours} hours")

        return filtered_entries

    def _process_dataset(self, dataset: DatasetMeta, min_duration: float, max_duration: float):
        entries = read_manifest(dataset.manifest_path)
        entries = self._filter_by_duration(
            entries=entries,
            min_duration=min_duration,
            max_duration=max_duration
        )

        samples = []
        sample_weights = []
        for entry in entries:
            audio_path = entry["audio_filepath"]

            if "normalized_text" in entry:
                text = entry["normalized_text"]
            else:
                text = entry["text"]

            if self.speaker_index_map:
                speaker = entry["speaker"]
            else:
                speaker = None

            _, audio_rel_path = get_audio_paths(audio_path=audio_path, base_path=dataset.audio_dir)

            sample = DatasetSample(
                audio_dir=dataset.audio_dir,
                audio_rel_path=audio_rel_path,
                sup_data_dir=dataset.sup_data_dir,
                text=text,
                speaker=speaker
            )
            samples.append(sample)
            sample_weights.append(dataset.sample_weight)

        return samples, sample_weights

    def __getitem__(self, index):
        data = self.data_samples[index]

        audio_path = data.audio_dir / data.audio_rel_path

        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        example = TTSTrainingExample(audio=audio, text=data.text)

        if data.speaker:
            example.speaker_index = self.speaker_index_map[data.speaker]

        sup_data_file_name = get_sup_data_file_name(data.audio_rel_path)
        for data_type in self.sup_data_types:
            sup_data_path = data.sup_data_dir / data_type / sup_data_file_name
            sup_data_value = torch.load(sup_data_path)
            setattr(example, data_type, sup_data_value)

        for data_type, feature_processors in self.data_processors:
            for processor in feature_processors:
                processor.process(training_example=example, field=data_type)

        return example

    def collate_fn(self, batch):
        return