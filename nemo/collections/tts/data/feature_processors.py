import torch
import os
import json
from pathlib import Path
from nemo.collections.tts.data.tts_multi_dataset import FeatureProcessor, TTSTrainingExample


class FeatureScaler(FeatureProcessor):

    def __init__(self, add_value: float = 0.0, div_value: float = 1.0):
        self.add_value = add_value
        self.div_value = div_value

    def process(self, training_example: TTSTrainingExample, field: str) -> None:
        feature = getattr(training_example, field)
        feature = (feature + self.add_value) / self.div_value
        setattr(training_example, field, feature)


class LogCompression(FeatureProcessor):

    def __init__(self, log_zero_guard_type: str = "add", log_zero_guard_value: int = 1.0):
        if log_zero_guard_type == "add":
            self.guard_fn = self._add_guard
        elif log_zero_guard_type == "clamp":
            self.guard_fn = self._clamp_guard
        else:
            raise ValueError(f"Unsupported log zero guard type: '{log_zero_guard_type}'")

        self.guard_type = log_zero_guard_type
        self.guard_value = log_zero_guard_value

    def _add_guard(self, feature: torch.Tensor):
        return feature + self.guard_value

    def _clamp_guard(self, feature: torch.Tensor):
        return torch.clamp(feature, min=self.guard_value)

    def process(self, training_example: TTSTrainingExample, field: str) -> None:
        feature = getattr(training_example, field)

        feature = self.guard_fn(feature)
        feature = torch.log(feature)

        setattr(training_example, field, feature)


class MeanVarianceNormalization(FeatureProcessor):

    def __init__(self, stats_path: Path, feature_name: str, mask_voiced: bool = True):
        self.mask_voiced = mask_voiced

        if not os.path.exists(stats_path):
            raise ValueError(f"Statistics file does not exist: {stats_path}")

        with open(Path(stats_path), 'r', encoding="utf-8") as pitch_f:
            stats_dict = json.load(pitch_f)
            self.mean = stats_dict["default"][f"{feature_name}_mean"]
            self.std = stats_dict["default"][f"{feature_name}_std"]

    def process(self, training_example: TTSTrainingExample, field: str) -> None:
        feature = getattr(training_example, field)

        feature = (feature - self.mean) / self.std
        if self.mask_voiced:
            feature = feature[training_example.voiced]

        setattr(training_example, field, feature)


class MeanVarianceSpeakerNormalization(FeatureProcessor):

    def __init__(self, stats_path: Path, feature_name: str, mask_voiced: bool = True, fallback_to_default: bool = False):
        self.key_mean = f"{feature_name}_mean"
        self.key_std = f"{feature_name}_std"
        self.mask_voiced = mask_voiced
        self.fallback_to_default = fallback_to_default

        if not os.path.exists(stats_path):
            raise ValueError(f"Statistics file does not exist: {stats_path}")

        with open(Path(stats_path), 'r', encoding="utf-8") as pitch_f:
            self.stats_dict = json.load(pitch_f)

    def process(self, training_example: TTSTrainingExample, field: str) -> None:
        feature = getattr(training_example, field)

        speaker = training_example.speaker
        if speaker in self.stats_dict:
            stats = self.stats_dict[speaker]
        elif self.fallback_to_default:
            stats = self.stats_dict["default"]
        else:
            raise ValueError(f"Statistics not found for speaker: {speaker}")

        feature_mean = stats[self.key_mean]
        feature_std = stats[self.key_std]

        feature = (feature - feature_mean) / feature_std

        if self.mask_voiced:
            feature = feature[training_example.voiced]

        setattr(training_example, field, feature)