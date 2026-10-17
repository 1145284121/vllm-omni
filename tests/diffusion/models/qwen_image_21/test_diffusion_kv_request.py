# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.lora.request import LoRARequest
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

import vllm_omni.diffusion.diffusion_engine as engine_module
import vllm_omni.diffusion.diffusion_kv.kv_cache_utils as kv_utils
import vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 as pipeline_module
import vllm_omni.diffusion.models.qwen_image_21.request_layout as layout_module
from tests.helpers.kv_layout import build_kv_cache_tensor
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    get_qwen_image_21_pre_process_func,
)
from vllm_omni.diffusion.models.qwen_image_21.request_layout import (
    PROMPT_TEMPLATE_T2I,
    SYS_PROMPT,
    QwenImage21LayoutTokenizer,
    QwenImage21PreparedLayout,
    build_qwen_image_21_diffusion_kv_requests,
    native_kv_covers_cond_images,
    prepare_qwen_image_21_layout,
    prepare_qwen_image_21_prefix_cache,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeCharTokenizer:
    """Deterministic character-level tokenizer with a single token for <|image_pad|>."""

    IMG_PAD_ID = 9999

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        ids: list[int] = []
        idx = 0
        marker = "<|image_pad|>"
        while idx < len(text):
            if text.startswith(marker, idx):
                ids.append(self.IMG_PAD_ID)
                idx += len(marker)
            else:
                ids.append(ord(text[idx]) + 10)
                idx += 1
        return {"input_ids": ids}


def _layout_tokenizer() -> QwenImage21LayoutTokenizer:
    return QwenImage21LayoutTokenizer(_FakeCharTokenizer())


def _make_ref_payloads(num_images: int = 2, *, fill: float = 0.25):
    vae_images = [torch.full((1, 4, 1, 64, 64), fill + i, dtype=torch.float32) for i in range(num_images)]
    input_image_sizes = [(64, 64) for _ in range(num_images)]
    prompt_images = [PIL.Image.new("RGB", (64, 64), color=(int(50 + i * 20), 60, 70)) for i in range(num_images)]
    return {
        "prompt_image": prompt_images,
        "vae_images": vae_images,
        "input_image_sizes": input_image_sizes,
    }


def _make_request(
    *,
    prompt: str = "restyle card one with card two",
    negative_prompt: str | None = None,
    true_cfg_scale: float = 1.0,
    num_images: int = 2,
    request_id: str = "req-1",
    seed: int | None = 42,
    height: int = 64,
    width: int = 64,
    fill: float = 0.25,
) -> OmniDiffusionRequest:
    prompt_dict: dict = {
        "prompt": prompt,
        "additional_information": _make_ref_payloads(num_images, fill=fill) if num_images > 0 else {},
    }
    if negative_prompt is not None:
        prompt_dict["negative_prompt"] = negative_prompt
    return OmniDiffusionRequest(
        prompt=prompt_dict,
        sampling_params=OmniDiffusionSamplingParams(
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=4,
            seed=seed,
        ),
        request_id=request_id,
    )


def test_dense_legacy_preprocess_does_not_initialize_layout_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: pytest.fail("dense_legacy must not initialize QwenImage21LayoutTokenizer"),
    )

    get_qwen_image_21_pre_process_func(
        SimpleNamespace(model="model", diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    )


def test_paged_preprocess_attaches_layout_and_scheduler_kv_requests(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: _layout_tokenizer(),
    )
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(
            model="model",
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
        )
    )
    req = OmniDiffusionRequest(
        prompt={
            "prompt": "make the yacht white",
            "negative_prompt": "blurry",
            "multi_modal_data": {"image": [PIL.Image.new("RGB", (512, 512))]},
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            true_cfg_scale=4.0,
            num_inference_steps=4,
            seed=7,
        ),
        request_id="edit-req",
    )

    prepared_req = preprocess(req)

    assert prepared_req is req
    assert isinstance(req.prepared_layout, QwenImage21PreparedLayout)
    assert req.prepared_layout.num_branches == 2
    assert req.diffusion_kv_requests is not None
    assert len(req.diffusion_kv_requests) == 2
    assert req.prepared_layout.target_len == 16
    assert [r.sequence_id for r in req.diffusion_kv_requests] == [0, 1]


def test_paged_preprocess_expands_startup_kv_profile_request(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: _layout_tokenizer(),
    )
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(
            model="model",
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
        )
    )
    profile_req = OmniDiffusionRequest(
        prompt={"prompt": "dummy run"},
        sampling_params=OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=1,
        ),
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
    )

    preprocess(profile_req)

    assert isinstance(profile_req.prepared_layout, QwenImage21PreparedLayout)
    assert len(profile_req.prepared_layout.input_image_sizes) == 4
    assert profile_req.prepared_layout.num_branches == 2
    assert profile_req.diffusion_kv_requests is not None
    assert len(profile_req.diffusion_kv_requests) == 2
    assert profile_req.diffusion_kv_requests[0].prefix_len > 4 * 4096


def test_system_prefix_dropped_and_images_expanded_4x() -> None:
    tok = _layout_tokenizer()
    expected_drop = len(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n")
    assert tok.drop_idx == expected_drop
    assert tok.baseline_len == len(PROMPT_TEMPLATE_T2I.format(""))

    req = _make_request(prompt="hello", num_images=2, height=64, width=64)
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)

    assert layout.num_branches == 1
    branch = layout.branches[0]
    # Each 64x64 image has (64//16)*(64//16) = 16 latent tokens (4 VLM slots).
    assert len(branch.image_spans) == 2
    assert [span.stop - span.start for span, _ in branch.image_spans] == [16, 16]
    assert sum(branch.vlm_image_pad_mask) == 8


@pytest.fixture
def prefix_cache_manager(request):
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    num_blocks = 128
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[build_kv_cache_tensor(spec, num_blocks, ["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    manager = DiffusionKVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_prefix_caching=getattr(request, "param", True),
    )
    yield manager
    manager.close()


@pytest.mark.parametrize("prefix_cache_manager", [False], indirect=True)
@pytest.mark.parametrize("true_cfg_scale", [1.0, 4.0])
def test_paged_cache_off_prepares_and_allocates_without_identity_work(
    monkeypatch, prefix_cache_manager, true_cfg_scale
) -> None:
    tok = _layout_tokenizer()
    config = SimpleNamespace(
        model="fake",
        model_class_name="QwenImage21Pipeline",
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
    )
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(layout_module, "QwenImage21LayoutTokenizer", lambda *_args, **_kwargs: tok)
    monkeypatch.setattr(engine_module, "get_diffusion_post_process_func", lambda _config: None)
    fail = Mock(side_effect=AssertionError("disabled prefix cache must not prepare identity"))
    monkeypatch.setattr(engine_module, "get_diffusion_prefix_cache_func", fail)
    monkeypatch.setattr(layout_module, "prepare_qwen_image_21_prefix_cache", fail)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", fail)
    monkeypatch.setattr(kv_utils, "hash_prefix_cache_value", fail)

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = config
    engine._init_process_hooks(config)
    engine.prefix_cache_func = fail

    req = _make_request(
        prompt="edit",
        negative_prompt="blurry" if true_cfg_scale > 1.0 else None,
        true_cfg_scale=true_cfg_scale,
    )
    engine._prepare_request_for_admission(req)
    assert isinstance(req.prepared_layout, QwenImage21PreparedLayout)
    assert all(row.cache_token_ids == () and row.mm_features == [] for row in req.diffusion_kv_requests)
    metadata = prefix_cache_manager.reserve_request(req.request_id, req.diffusion_kv_requests)
    assert metadata is not None
    assert all(row.cached_prefix_len == 0 for row in metadata.sequences)
    prefix_cache_manager.publish_request(req.request_id)
    prefix_cache_manager.free_request(req.request_id)
    fail.assert_not_called()


def test_reference_content_is_hashed_once_across_cfg_rows(monkeypatch) -> None:
    tok = _layout_tokenizer()
    hasher = Mock(wraps=layout_module.hash_prefix_cache_value)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", hasher)

    req = _make_request(
        prompt="restyle card",
        negative_prompt="blurry",
        true_cfg_scale=4.0,
        num_images=2,
    )
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)
    req.prepared_layout = layout
    req.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(req, layout)
    prepare_qwen_image_21_prefix_cache(req)

    # 2 reference images hashed once total across both CFG branches, while
    # positive and negative branches use isolated cache namespaces.
    assert sum(call.args[0][0] == "qwen-image-21-reference-v1" for call in hasher.call_args_list) == 2
    assert len(req.diffusion_kv_requests[0].mm_features) == 2
    assert req.diffusion_kv_requests[0].mm_features == req.diffusion_kv_requests[1].mm_features
    assert req.diffusion_kv_requests[0].cache_namespace != req.diffusion_kv_requests[1].cache_namespace
    assert req.diffusion_kv_requests[0].cache_namespace == kv_utils.get_cache_namespace(
        "qwen-image-21-primary-v1-cond", req.sampling_params
    )
    assert req.diffusion_kv_requests[1].cache_namespace == kv_utils.get_cache_namespace(
        "qwen-image-21-primary-v1-uncond", req.sampling_params
    )


def test_prefix_lookup_hits_full_partial_image_and_isolates_lora(
    prefix_cache_manager: DiffusionKVCacheManager,
) -> None:
    tok = _layout_tokenizer()

    def build(
        request_id: str,
        *,
        prompt: str = "replace the logo with AnchorCache",
        seed: int = 42,
        change_img2: bool = False,
        lora_id: int | None = None,
        lora_scale: float = 1.0,
    ):
        req = _make_request(
            prompt=prompt,
            negative_prompt="blurry",
            true_cfg_scale=4.0,
            num_images=2,
            request_id=request_id,
            seed=seed,
        )
        if change_img2:
            req.prompt["additional_information"]["vae_images"][1].fill_(9.0)
        if lora_id is not None:
            req.sampling_params.lora_request = LoRARequest(
                lora_name=f"adapter-{lora_id}",
                lora_int_id=lora_id,
                lora_path=f"/adapters/{lora_id}",
            )
            req.sampling_params.lora_scale = lora_scale
        layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)
        req.prepared_layout = layout
        req.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(req, layout)
        prepare_qwen_image_21_prefix_cache(req)
        return req

    cold_req = build("cold")
    cold_meta = prefix_cache_manager.reserve_request("cold", cold_req.diffusion_kv_requests)
    assert cold_meta is not None
    assert all(seq.cached_prefix_len == 0 for seq in cold_meta.sequences)
    prefix_cache_manager.publish_request("cold")
    prefix_cache_manager.free_request("cold")

    # 1. Different seed with identical images and promptstill hits full common prefix.
    same_seed_diff = build("same-diff-seed", seed=999)
    meta_same = prefix_cache_manager.reserve_request("same-diff-seed", same_seed_diff.diffusion_kv_requests)
    assert meta_same is not None
    image_prefix_end = cold_req.prepared_layout.branches[0].image_prefix_end
    assert meta_same.sequences[0].cached_prefix_len >= image_prefix_end
    assert native_kv_covers_cond_images(same_seed_diff.prepared_layout, meta_same.sequences[0].cached_prefix_len)
    prefix_cache_manager.free_request("same-diff-seed")

    # 2. Edited instruction hits both reference images and covers cond images.
    edited = build("edited-prompt", prompt="turn the background dark blue")
    meta_edited = prefix_cache_manager.reserve_request("edited-prompt", edited.diffusion_kv_requests)
    assert meta_edited is not None
    assert meta_edited.sequences[0].cached_prefix_len >= image_prefix_end
    assert native_kv_covers_cond_images(edited.prepared_layout, meta_edited.sequences[0].cached_prefix_len)
    prefix_cache_manager.free_request("edited-prompt")

    # 3. Changing only the second image still reuses the first image's span.
    img2_changed = build("img2-changed", change_img2=True)
    meta_img2 = prefix_cache_manager.reserve_request("img2-changed", img2_changed.diffusion_kv_requests)
    assert meta_img2 is not None
    first_img_end = cold_req.prepared_layout.branches[0].image_spans[0][0].stop
    assert first_img_end <= meta_img2.sequences[0].cached_prefix_len < image_prefix_end
    assert not native_kv_covers_cond_images(img2_changed.prepared_layout, meta_img2.sequences[0].cached_prefix_len)
    prefix_cache_manager.free_request("img2-changed")

    # 4. Different LoRA adapter misses completely.
    lora_changed = build("lora-changed", lora_id=1, lora_scale=0.8)
    meta_lora = prefix_cache_manager.reserve_request("lora-changed", lora_changed.diffusion_kv_requests)
    assert meta_lora is not None
    assert all(seq.cached_prefix_len == 0 for seq in meta_lora.sequences)
    prefix_cache_manager.free_request("lora-changed")

    # 5. Swapping positive and negative prompts in T2I does not cross-hit across
    # cond/uncond branches (preventing uninitialized cross-rank reads under CFGP2).
    t2i_a = _make_request(
        prompt="bright studio portrait",
        negative_prompt="dark blurry grain",
        true_cfg_scale=4.0,
        num_images=0,
        request_id="t2i-a",
    )
    t2i_a.prepared_layout = prepare_qwen_image_21_layout(t2i_a, layout_tokenizer=tok, vae_scale_factor=16)
    t2i_a.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(t2i_a, t2i_a.prepared_layout)
    prepare_qwen_image_21_prefix_cache(t2i_a)
    meta_t2i_a = prefix_cache_manager.reserve_request("t2i-a", t2i_a.diffusion_kv_requests)
    assert meta_t2i_a is not None
    prefix_cache_manager.publish_request("t2i-a")
    prefix_cache_manager.free_request("t2i-a")

    t2i_swapped = _make_request(
        prompt="dark blurry grain",
        negative_prompt="bright studio portrait",
        true_cfg_scale=4.0,
        num_images=0,
        request_id="t2i-swapped",
    )
    t2i_swapped.prepared_layout = prepare_qwen_image_21_layout(t2i_swapped, layout_tokenizer=tok, vae_scale_factor=16)
    t2i_swapped.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(
        t2i_swapped, t2i_swapped.prepared_layout
    )
    prepare_qwen_image_21_prefix_cache(t2i_swapped)
    meta_t2i_swapped = prefix_cache_manager.reserve_request("t2i-swapped", t2i_swapped.diffusion_kv_requests)
    assert meta_t2i_swapped is not None
    # Only the shared user header ("<|im_start|>user\n", 17 chars -> 4 blocks = 16 tokens)
    # may match within each branch's own namespace; the swapped prompt body must not cross-hit.
    assert all(seq.cached_prefix_len < seq.prefix_len for seq in meta_t2i_swapped.sequences)
    prefix_cache_manager.free_request("t2i-swapped")


def test_sequential_cfg_prepares_single_branch_batches_without_two_row_prefill() -> None:
    from contextlib import contextmanager

    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
        DiffusionPagedAttentionMetadata,
        DiffusionPagedAttentionRow,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import (
        QwenImage21CFGParallelMixin,
        _set_denoise_step_idx,
    )

    row0 = DiffusionPagedAttentionRow(request_id="r", sequence_id=0, kv_start_pos=0, query_len=20518, seq_len=20518)
    row1 = DiffusionPagedAttentionRow(request_id="r", sequence_id=1, kv_start_pos=0, query_len=20517, seq_len=20517)
    metadata = DiffusionPagedAttentionMetadata(prefill_rows=(row0, row1), denoise_rows=(row0, row1))
    prepared_batches: list[tuple[int, ...]] = []

    class _FakeAdapter:
        def prepare_batch(self, rows):
            num_tokens = sum(r.query_len for r in rows)
            if num_tokens > 32768:
                raise ValueError(f"Paged attention batch has {num_tokens} tokens; Worker capacity is 32768")
            seq_ids = tuple(r.sequence_id for r in rows)
            prepared_batches.append(seq_ids)
            return seq_ids

        @contextmanager
        def activate(self, batch):
            yield batch

    class _FakeRuntime:
        def __init__(self) -> None:
            self.metadata = metadata
            self.adapter = _FakeAdapter()

        def ensure_active(self, step_idx):
            rows = self.metadata.rows_for_step(step_idx)
            return self.adapter.prepare_batch(rows)

        def _deactivate(self) -> None:
            return None

    class _FakePipe(QwenImage21CFGParallelMixin):
        def predict_noise(self, **kwargs):
            return kwargs["out"]

        def combine_cfg_noise(self, pos, neg, scale, normalize):
            del scale, normalize
            return pos[0] + neg[0]

    pipe = _FakePipe()
    runtime = _FakeRuntime()
    with set_forward_context(vllm_config=None, omni_diffusion_config=None, paged_kv_runtime=runtime):
        _set_denoise_step_idx(0, sequential_cfg=True)
        out = pipe.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=4.0,
            positive_kwargs={"out": torch.ones(1, 4, 2)},
            negative_kwargs={"out": torch.ones(1, 4, 2)},
            output_slice=2,
        )
    assert out.shape == (1, 2, 2)
    assert prepared_batches == [(0,), (1,)]


def test_prepare_generation_context_skips_vae_and_prompt_encode_on_prefix_hit() -> None:
    tok = _layout_tokenizer()
    req = _make_request(prompt="keep card", num_images=2, height=64, width=64)
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)

    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.vae_scale_factor = 16
    pipe.latent_channels = 4
    pipe._max_length = 8192
    pipe.device = torch.device("cpu")
    pipe.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        parallel_config=SimpleNamespace(cfg_parallel_size=1),
    )
    pipe.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipe.transformer = SimpleNamespace(in_channels=4)
    pipe.encode_prompt = Mock(
        return_value=(
            torch.zeros(1, 20, 8),
            torch.ones(1, 20, dtype=torch.long),
            torch.tensor([layout.branches[0].vlm_image_pad_mask], dtype=torch.bool),
        )
    )
    pipe.prepare_latents = Mock(return_value=(torch.zeros(1, 16, 4), None))
    pipe.prepare_timesteps = Mock(return_value=(torch.tensor([1000.0, 500.0]), 2))
    per_request_images = pipe._extract_request_images([req.prompt])

    # Partial hit covering all condition images: skips condition VAE, runs encode_prompt.
    image_end = layout.branches[0].image_prefix_end
    pipe._prepare_generation_context(
        prompt=["keep card"],
        negative_prompt=None,
        height=64,
        width=64,
        num_inference_steps=2,
        sigmas=None,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(0),
        true_cfg_scale=1.0,
        max_sequence_length=8192,
        per_request_images=per_request_images,
        prepared_layout=layout,
        paged_kv_cached_prefix_len=image_end,
    )
    assert pipe.encode_prompt.call_count == 1
    assert pipe.prepare_latents.call_args.args[0] is None

    # Full prefix hit: skips both encode_prompt and condition VAE.
    pipe.encode_prompt.reset_mock()
    pipe.prepare_latents.reset_mock()
    ctx = pipe._prepare_generation_context(
        prompt=["keep card"],
        negative_prompt=None,
        height=64,
        width=64,
        num_inference_steps=2,
        sigmas=None,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(0),
        true_cfg_scale=1.0,
        max_sequence_length=8192,
        per_request_images=per_request_images,
        prepared_layout=layout,
        paged_kv_cached_prefix_len=layout.branches[0].prefix_len,
    )
    assert pipe.encode_prompt.call_count == 0
    assert pipe.prepare_latents.call_args.args[0] is None
    assert ctx["prompt_embeds"] is None

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _paged_kv_fp32_matmul_reduction,
    )

    prev = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    with _paged_kv_fp32_matmul_reduction(True):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
    assert bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction) is prev
