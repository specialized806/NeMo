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

import pytest
from torch import nn

from nemo.lightning.io.state import StateDictTransform, TransformCTX, state_transform


class TestStateDictTransform:
    """
    Tests for the StateDictTransform functionality.
    """

    @pytest.fixture
    def mock_ctx(self):
        """
        Provides a mock transformation context with predefined source and target states.

        Returns
        -------
            TransformCTX: A context object with source and target states.
        """
        source_state = {
            "model.layers.0.self_attn.q_proj.weight": 1,
            "model.layers.0.self_attn.k_proj.weight": 2,
            "model.layers.0.self_attn.v_proj.weight": 3,
            "model.layers.0.mlp.experts.0.gate_proj.weight": 4,
            "model.layers.0.mlp.experts.0.up_proj.weight": 5,
            "model.layers.0.mlp.experts.0.down_proj.weight": 8,
            "model.layers.0.mlp.experts.1.gate_proj.weight": 6,
            "model.layers.0.mlp.experts.1.up_proj.weight": 7,
            "model.layers.0.mlp.experts.1.down_proj.weight": 8,
            "model.layers.1.self_attn.q_proj.weight": 2,
            "model.layers.1.self_attn.k_proj.weight": 3,
            "model.layers.1.self_attn.v_proj.weight": 4,
            "model.layers.1.mlp.experts.0.gate_proj.weight": 5,
            "model.layers.1.mlp.experts.0.up_proj.weight": 6,
            "model.layers.1.mlp.experts.0.down_proj.weight": 9,
            "model.layers.1.mlp.experts.1.gate_proj.weight": 7,
            "model.layers.1.mlp.experts.1.up_proj.weight": 8,
            "model.layers.1.mlp.experts.1.down_proj.weight": 9,
        }
        target_state = {
            "decoder.layers.0.self_attention.linear_qkv.weight": -1,
            "decoder.layers.0.self_attention.linear_proj.weight": -1,
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": -1,
            "decoder.layers.0.mlp.experts.linear_fc1.weight1": -1,
            "decoder.layers.0.mlp.experts.linear_fc2.weight": -1,
            "decoder.layers.1.self_attention.linear_qkv.weight": -1,
            "decoder.layers.1.self_attention.linear_proj.weight": -1,
            "decoder.layers.1.mlp.experts.linear_fc1.weight0": -1,
            "decoder.layers.1.mlp.experts.linear_fc1.weight1": -1,
            "decoder.layers.1.mlp.experts.linear_fc2.weight": -1,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    @pytest.fixture
    def mock_multi_target_ctx(self):
        """
        Provides a mock transformation context with a source state that matches the expected source_key
        and a target state prepared with initial values for the expected target_keys.
        """
        source_state = {
            "decoder.layers.0.mlp.linear_fc1.weight": 1,
            "decoder.layers.1.mlp.linear_fc1.weight": 2,
            "decoder.layers.2.mlp.experts.linear_fc1.weight0": 23,
            "decoder.layers.2.mlp.experts.linear_fc1.weight1": 45,
            "decoder.layers.3.mlp.experts.linear_fc1.weight0": 34,
            "decoder.layers.3.mlp.experts.linear_fc1.weight1": 56,
        }
        # Populate target_state with initial placeholder values for keys expected to be matched and updated
        target_state = {
            "model.layers.0.mlp.gate_proj.weight": -1,
            "model.layers.0.mlp.up_proj.weight": -1,
            "model.layers.1.mlp.gate_proj.weight": -1,
            "model.layers.1.mlp.up_proj.weight": -1,
            "model.layers.2.mlp.experts.0.gate_proj.weight": -1,
            "model.layers.2.mlp.experts.0.up_proj.weight": -1,
            "model.layers.2.mlp.experts.1.gate_proj.weight": -1,
            "model.layers.2.mlp.experts.1.up_proj.weight": -1,
            "model.layers.3.mlp.experts.0.gate_proj.weight": -1,
            "model.layers.3.mlp.experts.0.up_proj.weight": -1,
            "model.layers.3.mlp.experts.1.gate_proj.weight": -1,
            "model.layers.3.mlp.experts.1.up_proj.weight": -1,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    def test_transform_with_single_source_single_target(self, mock_ctx):
        """
        Test transformation when a single source and target key is specified.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.mlp.experts.0.down_proj.weight",
            target_key="decoder.layers.*.mlp.experts.linear_fc2.weight",
            transform=lambda ctx, x: x * 100,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.mlp.experts.linear_fc2.weight"] == 800
        assert mock_ctx.target_state["decoder.layers.1.mlp.experts.linear_fc2.weight"] == 900

    def test_transform_with_multiple_sources(self, mock_ctx):
        """
        Test transformation when multiple source keys are specified.
        """
        transform = StateDictTransform(
            source_key=(
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ),
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, q, k, v: q * 100 + k * 10 + v,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 123
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 234

    def test_transform_with_multiple_mapped_sources(self, mock_ctx):
        """
        Test transformation with a dictionary mapping for source keys.
        """
        transform = StateDictTransform(
            source_key={
                "q": "model.layers.*.self_attn.q_proj.weight",
                "k": "model.layers.*.self_attn.k_proj.weight",
                "v": "model.layers.*.self_attn.v_proj.weight",
            },
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, q, k, v: q * 100 + k * 10 + v,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 123
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 234

    def test_transform_with_variable_arguments(self, mock_ctx):
        """
        Test transformation with a wildcard pattern and variable arguments.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.self_attn.*_proj.weight",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, *args: sum(args),
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 6
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 9

    def test_transform_with_no_matching_source(self, mock_ctx):
        """
        Test transformation when no source keys match the pattern.
        """
        transform = StateDictTransform(
            source_key="non.existent.pattern",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, *args: sum(args),
        )
        with pytest.raises(ValueError):
            transform(mock_ctx)

    def test_transform_with_multiple_targets(self, mock_multi_target_ctx):
        """
        Test transformation where the target_key is a tuple and the transform function
        returns multiple values that are then unrolled to these target keys.
        """

        # Define a transformation that splits the input into two parts
        def split_transform(ctx, x):
            return x - 1, x + 1

        # Apply the transformation
        transform = StateDictTransform(
            source_key="decoder.layers.*.mlp.linear_fc1.weight",
            target_key=(
                "model.layers.*.mlp.gate_proj.weight",
                "model.layers.*.mlp.up_proj.weight",
            ),
            transform=split_transform,
        )
        transform(mock_multi_target_ctx)

        # Check that the target state has been updated correctly
        assert mock_multi_target_ctx.target_state["model.layers.0.mlp.gate_proj.weight"] == 0
        assert mock_multi_target_ctx.target_state["model.layers.0.mlp.up_proj.weight"] == 2
        assert mock_multi_target_ctx.target_state["model.layers.1.mlp.gate_proj.weight"] == 1
        assert mock_multi_target_ctx.target_state["model.layers.1.mlp.up_proj.weight"] == 3

    def test_transform_with_multiple_sources_multiple_wildcards(self, mock_ctx):
        """
        Test transformation when multiple source keys are specified, each with more than 1 wildcard.
        """
        transform = StateDictTransform(
            source_key=(
                "model.layers.*.mlp.experts.*.gate_proj.weight",
                "model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            target_key="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            transform=lambda ctx, gate, up: gate * 10 + up,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.mlp.experts.linear_fc1.weight0"] == 45
        assert mock_ctx.target_state["decoder.layers.0.mlp.experts.linear_fc1.weight1"] == 67
        assert mock_ctx.target_state["decoder.layers.1.mlp.experts.linear_fc1.weight0"] == 56
        assert mock_ctx.target_state["decoder.layers.1.mlp.experts.linear_fc1.weight1"] == 78

    def test_transform_with_multiple_targets_multiple_wildcards(self, mock_multi_target_ctx):
        """
        Test transformation when multiple target keys are specified, each with more than 1 wildcard.
        """

        def split_transform(ctx, x):
            return x // 10, x % 10

        transform = StateDictTransform(
            source_key="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            target_key=(
                "model.layers.*.mlp.experts.*.gate_proj.weight",
                "model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            transform=split_transform,
        )

        transform(mock_multi_target_ctx)
        assert mock_multi_target_ctx.target_state["model.layers.2.mlp.experts.0.gate_proj.weight"] == 2
        assert mock_multi_target_ctx.target_state["model.layers.2.mlp.experts.0.up_proj.weight"] == 3
        assert mock_multi_target_ctx.target_state["model.layers.2.mlp.experts.1.gate_proj.weight"] == 4
        assert mock_multi_target_ctx.target_state["model.layers.2.mlp.experts.1.up_proj.weight"] == 5
        assert mock_multi_target_ctx.target_state["model.layers.3.mlp.experts.0.gate_proj.weight"] == 3
        assert mock_multi_target_ctx.target_state["model.layers.3.mlp.experts.0.up_proj.weight"] == 4
        assert mock_multi_target_ctx.target_state["model.layers.3.mlp.experts.1.gate_proj.weight"] == 5
        assert mock_multi_target_ctx.target_state["model.layers.3.mlp.experts.1.up_proj.weight"] == 6

    def test_transform_with_no_matching_target(self, mock_ctx):
        """
        Test transformation when no source keys match the pattern.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.mlp.experts.0.down_proj.weight",
            target_key="non.existent.pattern",
            transform=lambda ctx, *args: sum(args),
        )
        with pytest.raises(ValueError):
            transform(mock_ctx)

    def test_transform_with_invalid_transform_function(self, mock_ctx):
        """
        Test transformation with a transform function that does not match expected signature.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.self_attn.q_proj.weight",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx: 0,  # Invalid signature
        )
        with pytest.raises(ValueError):
            transform(mock_ctx)


class TestStateTransformDecorator:
    """
    Tests for the @state_transform decorator functionality.
    """

    @pytest.fixture
    def mock_ctx(self):
        """
        Provides a mock transformation context with predefined source and target states.
        """
        source_state = {
            'model.layers.1.self_attn.q_proj.weight': 1,
            'model.layers.1.self_attn.k_proj.weight': 2,
            'model.layers.1.self_attn.v_proj.weight': 3,
        }
        # Pre-populate target_state with initial values or placeholders
        target_state = {
            "decoder.layers.1.self_attention.linear_q.weight": 0,
            "decoder.layers.1.self_attention.linear_k.weight": 0,
            "decoder.layers.1.self_attention.linear_v.weight": 0,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    def test_single_transform(self, mock_ctx):
        """
        Test the @state_transform decorator with a single source and target key.
        """
        # Apply the transformation
        single_transform(mock_ctx)
        # Verify the target state is updated correctly
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_q.weight"] == 11

    def test_multiple_outputs_transform(self, mock_ctx):
        """
        Test the @state_transform decorator with a single source key and multiple target keys.
        """
        # Apply the transformation
        multiple_outputs_transform(mock_ctx)
        # Verify the target state is updated correctly for each key
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_q.weight"] == 1
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_k.weight"] == 2
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_v.weight"] == 3


@state_transform(
    source_key="model.layers.*.self_attn.q_proj.weight", target_key="decoder.layers.1.self_attention.linear_q.weight"
)
def single_transform(ctx, x):
    """
    A single transformation function that adds 10 to the input value.
    """
    return x + 10


@state_transform(
    source_key="model.layers.1.self_attn.*_proj.weight",
    target_key=("decoder.layers.1.self_attention.linear_*.weight",),
)
def multiple_outputs_transform(ctx, *args):
    """
    A transformation function that returns multiple values for multiple target keys.
    """
    return args
