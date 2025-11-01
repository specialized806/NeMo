# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.pipelines.buffered_ctc_pipeline import BufferedCTCPipeline
from nemo.collections.asr.inference.pipelines.buffered_rnnt_pipeline import BufferedRNNTPipeline
from nemo.collections.asr.inference.utils.enums import ASRDecodingType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging


class BufferedPipelineBuilder(BaseBuilder):
    """
    Buffered Pipeline Builder class.
    Builds the buffered CTC/RNNT/TDT pipelines.
    """

    @classmethod
    def build(cls, cfg: DictConfig) -> BufferedRNNTPipeline | BufferedCTCPipeline:
        """
        Build the buffered streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedRNNTPipeline or BufferedCTCPipeline object
        """
        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)

        if asr_decoding_type is ASRDecodingType.RNNT:
            return cls.build_buffered_rnnt_pipeline(cfg)
        elif asr_decoding_type is ASRDecodingType.CTC:
            return cls.build_buffered_ctc_pipeline(cfg)

        raise ValueError("Invalid asr decoding type for buffered streaming. Need to be one of ['CTC', 'RNNT']")

    @classmethod
    def get_rnnt_decoding_cfg(cls, cfg: DictConfig) -> RNNTDecodingConfig:
        """
        Get the decoding config for the RNNT pipeline.
        Returns:
            (RNNTDecodingConfig) Decoding config
        """
        decoding_cfg = RNNTDecodingConfig()

        # greedy_batch decoding strategy required for stateless streaming
        decoding_cfg.strategy = "greedy_batch"

        # required to compute the middle token for transducers.
        decoding_cfg.preserve_alignments = False

        # temporarily stop fused batch during inference.
        decoding_cfg.fused_batch_size = -1

        # return and write the best hypothesis only
        decoding_cfg.beam.return_best_hypothesis = True

        # setup ngram language model
        if hasattr(cfg.asr, "ngram_lm_model") and cfg.asr.ngram_lm_model != "":
            decoding_cfg.greedy.ngram_lm_model = cfg.asr.ngram_lm_model
            decoding_cfg.greedy.ngram_lm_alpha = cfg.asr.ngram_lm_alpha

        return decoding_cfg

    @classmethod
    def get_ctc_decoding_cfg(cls) -> CTCDecodingConfig:
        """
        Get the decoding config for the CTC pipeline.
        Returns:
            (CTCDecodingConfig) Decoding config
        """
        decoding_cfg = CTCDecodingConfig()
        decoding_cfg.strategy = "greedy"
        return decoding_cfg

    @classmethod
    def build_buffered_rnnt_pipeline(cls, cfg: DictConfig) -> BufferedRNNTPipeline:
        """
        Build the RNNT streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedRNNTPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_rnnt_decoding_cfg(cfg)
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building RNNT pipeline
        rnnt_pipeline = BufferedRNNTPipeline(cfg, asr_model, itn_model)
        logging.info(f"`{type(rnnt_pipeline).__name__}` pipeline loaded")
        return rnnt_pipeline

    @classmethod
    def build_buffered_ctc_pipeline(cls, cfg: DictConfig) -> BufferedCTCPipeline:
        """
        Build the CTC buffered streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedCTCPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_ctc_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building CTC pipeline
        ctc_pipeline = BufferedCTCPipeline(cfg, asr_model, itn_model)
        logging.info(f"`{type(ctc_pipeline).__name__}` pipeline loaded")
        return ctc_pipeline
