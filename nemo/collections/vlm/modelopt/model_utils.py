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

from functools import partial

import lightning.pytorch as L
import torch
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.llm.gpt.model.llama4_utils import get_llama4_layer_spec
from nemo.collections.vlm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.utils import logging


__all__ = ["set_modelopt_spec_if_exists_in_ckpt", "setup_trainer_and_restore_model_with_modelopt_spec"]


def set_modelopt_spec_if_exists_in_ckpt(model: L.LightningModule, path: str) -> None:
    """Set model.config.transformer_layer_spec to modelopt spec if modelopt_state exists in the checkpoint."""
    path = str(path).removeprefix("nemo://")  # Remove nemo:// prefix added by finetune_recipe
    modelopt_state_path = ckpt_to_weights_subdir(path, is_saving=False) / "modelopt_state"
    if not modelopt_state_path.exists() or hasattr(model, "module"):
        return

    if isinstance(model, (vlm.Llama4OmniModel)):
        _set_llama4_modelopt_spec(model.config)

        # Disable gradient accumulation fusion for QAT
        model.config.gradient_accumulation_fusion = False
    else:
        logging.warning(f"{type(model)} is not a Llama4OmniModel. Modelopt state will not be loaded.")


def _set_llama4_modelopt_spec(model_cfg: vlm.Llama4OmniConfig) -> vlm.Llama4OmniConfig:
    """
    Set the model layer spec to a modelopt spec variant. This function updates the model
    config with the appropriate modelopt layer specification based on the model type.

    Args:
        model_cfg (vlm.Llama4OmniConfig): The model config.

    Returns:
        vlm.Llama4OmniConfig: The model config updated for the modelopt layer specification.
    """
    logging.info("Setting model layer specification to the modelopt layer spec")

    if isinstance(model_cfg, vlm.Llama4OmniConfig):
        model_cfg.language_transformer_config.transformer_layer_spec = partial(
            _get_llama4_modelopt_spec, remap_te_layernorm=True
        )
    else:
        raise ValueError(f"No modelopt layer spec supported for config type {type(model_cfg)}")
    return model_cfg


def _get_llama4_modelopt_spec(
    config: TransformerConfig,
    local_core_attention: bool = False,
    remap_te_layernorm: bool = False,
    real_quant_cfg: str = "None",
    qk_l2_norm: bool = False,
) -> ModuleSpec:
    """Set model.config.transformer_layer_spec to llama4_modelopt_spec."""
    gpt_decoder_block_spec = get_gpt_modelopt_spec(
        config, local_core_attention, remap_te_layernorm, real_quant_cfg, qk_l2_norm
    )
    llama4_layer_spec = get_llama4_layer_spec(config, gpt_decoder_block_spec=gpt_decoder_block_spec)
    return llama4_layer_spec


def setup_trainer_and_restore_model_with_modelopt_spec(
    model_path: str,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    expert_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
    inference_only: bool = True,
    legacy_ckpt: bool = False,
    strategy_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
    model_config_overrides: dict | None = None,
) -> tuple[vlm.Llama4OmniModel, nl.Trainer]:
    """Loads a GPT model from a NeMo 2.0 checkpoint using modelopt layer spec.

    Args:
        model_path (str): Path to the NeMo checkpoint.
        tensor_model_parallel_size (int): Size of the tensor model parallelism.
        pipeline_model_parallel_size (int): Size of the pipeline model parallelism.
        num_layers_in_first_pipeline_stage (int): Number of layers in the first pipeline stage.
        num_layers_in_last_pipeline_stage (int): Number of layers in the last pipeline stage.
        expert_model_parallel_size (int): Size of the expert model parallelism.
        devices (int): Number of devices on each node.
        num_nodes (int): Number of nodes being used.
        inference_only (bool): If True, loads the model for inference only w/o initializing the optimizer.
        legacy_ckpt (bool): If True, allow loading ckpt saved with older version of TE.
        strategy_kwargs (Optional[dict]): Additional keyword arguments for nl.MegatronStrategy.
        trainer_kwargs (Optional[dict]): Additional keyword arguments for nl.Trainer.
        model_config_overrides (Optional[dict]): keyword arguments to override model config.

    Returns:
        Union[vlm.Llama4OmniModel]: The loaded model with the specified configuration.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}
    if trainer_kwargs is None:
        trainer_kwargs = {}
    if model_config_overrides is None:
        model_config_overrides = {}

    logging.info(f"Loading model from {model_path} with modelopt layer spec...")

    # TODO: setting ddp="pytorch" and deleting model.optim is a hackish way to disable DDP initialization.
    # Needs a systematic solution.
    if inference_only:
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_parallel_save_optim=False,
            setup_optimizers=False,
            ddp="pytorch",
            ckpt_load_strictness=StrictHandling.LOG_ALL if legacy_ckpt else None,
            **strategy_kwargs,
        )
    else:
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_strictness=StrictHandling.LOG_ALL if legacy_ckpt else None,
            **strategy_kwargs,
        )

    trainer = nl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed", params_dtype=torch.bfloat16, autocast_enabled=False, grad_reduce_in_fp32=True
        ),
        **trainer_kwargs,
    )

    model = nl.io.load_context(path=ckpt_to_context_subdir(model_path), subpath="model")

    _set_llama4_modelopt_spec(model.config)

    for k, v in model_config_overrides.items():
        logging.info(f"Overriding model.config.{k} to {v}")
        setattr(model.config, k, v)

    if inference_only:
        del model.optim
    if num_layers_in_first_pipeline_stage:
        model.config.num_layers_in_first_pipeline_stage = num_layers_in_first_pipeline_stage
    if num_layers_in_last_pipeline_stage:
        model.config.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage

    _setup_trainer_and_restore_model(model_path, trainer, model)
    trainer.strategy.restore_config = None  # No need to restore model weights again

    logging.info(f"Loaded model: {model}\n")
    return model, trainer
