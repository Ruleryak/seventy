"""GGUF (seventy/gguf.py, csrc/gguf.cu): the file reader on a small written file (metadata, tensors, split files),
the CPU dequantization of known blocks, the GPU dequantization of every format against the CPU one, and the
grouped expert product of the decode path against dense products."""

import struct

import numpy as np
import pytest
import torch

from seventy import gguf
from seventy.ops import kernels_available

GPU = torch.cuda.is_available() and kernels_available()
KINDS = ["Q4_0", "Q8_0", "Q2_0", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL",
         "IQ4_XS"]
IDS = {v: k for k, v in gguf.TYPE_NAMES.items()}


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, metadata: dict, tensors: list[tuple[str, tuple[int, ...], str, bytes]]):
    """A GGUF v3 file: string and uint32 metadata, tensors (name, dims fastest first, type, raw bytes)."""
    head = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        head += _string(k) + (struct.pack("<I", 8) + _string(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    offset = 0
    for name, dims, kind, raw in tensors:
        head += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", IDS[kind], offset)
        offset += -(-len(raw) // 32) * 32
    head += b"\0" * (-len(head) % 32)
    body = b"".join(raw + b"\0" * (-len(raw) % 32) for _, _, _, raw in tensors)
    path.write_bytes(head + body)


def random_blocks(kind: str, count: int, seed: int) -> np.ndarray:
    """count random blocks of kind, with small finite FP16 scales."""
    rng = np.random.default_rng(seed)
    values, size = gguf.BLOCKS[kind]
    b = rng.integers(0, 256, (count, size), dtype=np.uint8)
    halves = [(208, 210)] if kind == "Q6_K" else [(0, 2), (2, 4)] if kind in ("Q4_K", "Q5_K") else [(0, 2)]
    for a, z in halves:
        b[:, a:z] = (rng.standard_normal(count).astype(np.float16) * 0.01).view(np.uint8).reshape(count, 2)
    return b


def test_reader_and_split_files(tmp_path):
    w = np.arange(64, dtype=np.float32).reshape(2, 32)
    q = np.zeros((2, 18), np.uint8)
    q[:, :2] = np.array([0.5], np.float16).view(np.uint8)
    q[:, 2:] = 0xE4
    write_gguf(tmp_path / "m-00001-of-00002.gguf", {"general.architecture": "qwen4exp", "split.count": 2},
               [("a.weight", (32, 2), "F32", w.tobytes())])
    write_gguf(tmp_path / "m-00002-of-00002.gguf", {"split.count": 2}, [("b.weight", (64, 2), "Q2_0", q.tobytes())])
    meta, tensors = gguf.open_model(tmp_path / "m-00001-of-00002.gguf")
    assert meta["general.architecture"] == "qwen4exp"
    assert tensors["a.weight"].shape == (2, 32) and np.array_equal(tensors["a.weight"].dequantize(), w)
    b = tensors["b.weight"].dequantize()
    assert b.shape == (2, 64) and np.array_equal(b[0], np.tile([-0.5, 0.0, 0.5, 1.0], 16))


def test_known_blocks():
    d = np.array([0.25], np.float16).view(np.uint8)
    nl = np.concatenate((d, np.arange(16, dtype=np.uint8) | (np.arange(16, dtype=np.uint8)[::-1] << 4)))[None]
    kv = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
    assert np.array_equal(gguf.dequantize(nl, "IQ4_NL")[0], 0.25 * np.concatenate((kv, kv[::-1])))
    q8 = np.concatenate((d, np.arange(-16, 16, dtype=np.int8).view(np.uint8)))[None]
    assert np.array_equal(gguf.dequantize(q8, "Q8_0")[0], 0.25 * np.arange(-16, 16, dtype=np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(not GPU, reason="needs a CUDA GPU and the kernel library")
@pytest.mark.parametrize("kind", KINDS)
def test_gpu_dequantization_matches_the_cpu(kind):
    from seventy.ops import kernels

    values, size = gguf.BLOCKS[kind]
    k = values * 4 if values < 256 else 512
    rows = random_blocks(kind, 37 * (k // values), hash(kind) % 1000).reshape(37, -1)
    want = gguf.dequantize(rows, kind)
    got = kernels.gguf_dequant(torch.from_numpy(rows).cuda(), kind, k).float().cpu().numpy()
    np.testing.assert_allclose(got, want.astype(np.float16).astype(np.float32), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not GPU, reason="needs a CUDA GPU and the kernel library")
@pytest.mark.parametrize("kind", ["IQ3_XXS", "IQ3_S", "IQ2_XS", "IQ4_NL", "Q2_0", "Q4_K", "Q6_K"])
@pytest.mark.parametrize("tokens", [1, 5, 40])
def test_grouped_expert_product_matches_dense(kind, tokens):
    from seventy.ops import kernels

    values, size = gguf.BLOCKS[kind]
    experts, n, k, top = 6, 96, 512, 2
    blocks = random_blocks(kind, experts * n * (k // values), tokens + len(kind)).reshape(experts, n, -1)
    w = torch.from_numpy(blocks).cuda()
    dense = torch.from_numpy(gguf.dequantize(blocks.reshape(experts * n, -1), kind).reshape(experts, n, k)).cuda()
    g = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, k, generator=g).half().cuda()
    ids = torch.stack([torch.randperm(experts, generator=g)[:top] for _ in range(tokens)]).int().cuda()
    pw = torch.rand(tokens * top, generator=g).cuda()
    plan = kernels.moe_plan(ids.reshape(-1), 0, experts, 8)
    out = torch.zeros(tokens * top, n + 8, dtype=torch.float16, device="cuda")
    kernels.gguf_moe(x, top, w, kind, k, plan, out)
    y = torch.zeros(tokens * top, n, dtype=torch.float32, device="cuda")
    kernels.gguf_moe(x, top, w, kind, k, plan, y, pair_weight=pw)
    want = torch.stack([dense[int(ids.reshape(-1)[p])].float() @ x[p // top].float() for p in range(tokens * top)])
    torch.testing.assert_close(out[:, :n].float(), want, rtol=2e-2, atol=2e-2 * float(want.abs().max()))
    torch.testing.assert_close(y, want * pw[:, None], rtol=2e-2, atol=2e-2 * float(want.abs().max()))


def _write_meta(path, metadata: dict, tensors):
    """write_gguf with int arrays and uint32 values in the metadata."""
    head = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        head += _string(k)
        if isinstance(v, str):
            head += struct.pack("<I", 8) + _string(v)
        elif isinstance(v, list):
            head += struct.pack("<IIQ", 9, 11, len(v)) + b"".join(struct.pack("<q", x) for x in v)
        else:
            head += struct.pack("<II", 4, v)
    offset = 0
    for name, dims, kind, raw in tensors:
        head += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", IDS[kind], offset)
        offset += -(-len(raw) // 32) * 32
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"".join(raw + b"\0" * (-len(raw) % 32) for _, _, _, raw in tensors))


def test_checkpoint_undoes_the_converter(tmp_path):
    """Tensors in the Hugging Face layout, put through llama.cpp's qwen4exp converter changes (value heads
    tile-major, norms as 1 + w, -exp(A_log), the indexer projection in two parts, the shared gate as a vector, the
    convolution without its middle dimension), come back from GGUFCheckpoint as they were."""
    rng = np.random.default_rng(1)
    G, R, kd, vd, h = 2, 3, 4, 4, 8
    hf = {
        "linear_attn.A_log": rng.standard_normal(G * R).astype(np.float32),
        "linear_attn.dt_bias": rng.standard_normal(G * R).astype(np.float32),
        "linear_attn.in_proj_a.weight": rng.standard_normal((G * R, h)).astype(np.float32),
        "linear_attn.in_proj_z.weight": rng.standard_normal((G * R * vd, h)).astype(np.float32),
        "linear_attn.in_proj_qkv.weight": rng.standard_normal((2 * G * kd + G * R * vd, h)).astype(np.float32),
        "linear_attn.conv1d.weight": rng.standard_normal((2 * G * kd + G * R * vd, 1, 4)).astype(np.float32),
        "linear_attn.out_proj.weight": rng.standard_normal((h, G * R * vd)).astype(np.float32),
        "linear_attn.norm.weight": rng.standard_normal(vd).astype(np.float32),
        "attn_hyper_connection.hc_norm.weight": rng.standard_normal(h).astype(np.float32),
        "mlp.shared_expert_gate.weight": rng.standard_normal((1, h)).astype(np.float32),
        "self_attn.indexer.index_qk_proj.weight": rng.standard_normal((3 * 4, h)).astype(np.float32),
        "ple.norm_key.weight": rng.standard_normal(h).astype(np.float32),
    }

    def tile(x, axis, head):
        shape = list(x.shape)
        grouped = x.reshape(shape[:axis] + [G, R, head] + shape[axis + 1 :])
        return np.ascontiguousarray(np.swapaxes(grouped, axis, axis + 1)).reshape(shape)

    qk = 2 * G * kd
    raw = {
        "ssm_a": tile(-np.exp(hf["linear_attn.A_log"]), 0, 1),
        "ssm_dt.bias": tile(hf["linear_attn.dt_bias"], 0, 1),
        "ssm_alpha.weight": tile(hf["linear_attn.in_proj_a.weight"], 0, 1),
        "attn_gate.weight": tile(hf["linear_attn.in_proj_z.weight"], 0, vd),
        "attn_qkv.weight": np.concatenate((hf["linear_attn.in_proj_qkv.weight"][:qk],
                                           tile(hf["linear_attn.in_proj_qkv.weight"][qk:], 0, vd))),
        "ssm_conv1d.weight": np.concatenate((hf["linear_attn.conv1d.weight"][:qk, 0],
                                             tile(hf["linear_attn.conv1d.weight"][qk:, 0], 0, vd))),
        "ssm_out.weight": tile(hf["linear_attn.out_proj.weight"], 1, vd),
        "ssm_norm.weight": hf["linear_attn.norm.weight"],
        "hc_attn_norm.weight": hf["attn_hyper_connection.hc_norm.weight"] + 1,
        "ffn_gate_inp_shexp.weight": hf["mlp.shared_expert_gate.weight"][0],
        "indexer.q_proj.weight": hf["self_attn.indexer.index_qk_proj.weight"][:8],
        "indexer.k_proj.weight": hf["self_attn.indexer.index_qk_proj.weight"][8:],
        "ple_norm_key.weight": hf["ple.norm_key.weight"] + 1,
    }
    tensors = [(f"blk.0.{k}", tuple(reversed(v.shape)), "F32", v.astype(np.float32).tobytes()) for k, v in raw.items()]
    meta = {"general.architecture": "qwen4exp", "qwen4exp.ssm.group_count": G, "qwen4exp.ssm.time_step_rank": G * R,
            "qwen4exp.ssm.state_size": kd, "qwen4exp.ssm.inner_size": G * R * vd, "qwen4exp.block_count": 1,
            "qwen4exp.ple.layers": [0], "qwen4exp.ple.layer_multipliers": [3, 5, 7],
            "qwen4exp.ple.head_vocab_sizes": [11, 13], "qwen4exp.ple.head_offsets": [0, 11]}
    _write_meta(tmp_path / "m.gguf", meta, tensors)
    ckpt = gguf.GGUFCheckpoint(tmp_path)
    p = "model.language_model.layers.0."
    for name, want in hf.items():
        np.testing.assert_allclose(ckpt.tensor(p + name).numpy(), want, rtol=1e-6, atol=1e-6, err_msg=name)
    assert ckpt.tensor(p + "ple.ple_embedding.ngram_heads_offsets").tolist() == [0, 11]
    np.testing.assert_allclose(ckpt.linear_part(p + "linear_attn.in_proj_qkv", rows=(2, 5)).numpy(),
                               hf["linear_attn.in_proj_qkv.weight"][2:5], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ckpt.linear_part(p + "linear_attn.out_proj", rows=(0, h), cols=(4, 20)).numpy(),
                               hf["linear_attn.out_proj.weight"][:, 4:20], rtol=1e-6, atol=1e-6)


def test_checkpoint_reads_a_separate_mtp_file(tmp_path):
    """A separate MTP file (llama.cpp's nextn layer as the last block) in the folder gives the mtp.* tensors: eh_proj
    as fc_embedding and fc_hidden, the first hyper-connection products as their mix and injection rows, the norms
    without the + 1, the experts as GGUF blocks. has_mtp sees it; the main file is still the model."""
    from seventy.config import has_mtp

    rng = np.random.default_rng(2)
    h, rank, hc = 8, 4, 2
    main = [("blk.0.ssm_norm.weight", (4,), "F32", np.ones(4, np.float32).tobytes())]
    meta = {"general.architecture": "qwen4exp", "qwen4exp.ssm.group_count": 2, "qwen4exp.ssm.time_step_rank": 2,
            "qwen4exp.ssm.state_size": 4, "qwen4exp.ssm.inner_size": 8, "qwen4exp.block_count": 1}
    _write_meta(tmp_path / "m.gguf", meta, main)
    assert not has_mtp(tmp_path)
    eh = rng.standard_normal((h, 2 * h)).astype(np.float32)
    inject = rng.standard_normal((rank + hc, hc * h)).astype(np.float32)
    enorm = rng.standard_normal(h).astype(np.float32)
    q = rng.standard_normal((2 * h, h)).astype(np.float32)
    experts = random_blocks("Q8_0", 3 * h * 32 // 32, 3)
    raw = {"nextn.eh_proj.weight": (eh, "F32"), "hc_attn_down_inject.weight": (inject, "F32"),
           "nextn.enorm.weight": (enorm + 1, "F32"), "attn_q.weight": (q, "F32")}
    tensors = [(f"blk.1.{k}", tuple(reversed(v.shape)), t, v.tobytes()) for k, (v, t) in raw.items()]
    tensors.append(("blk.1.ffn_gate_exps.weight", (32, h, 3), "Q8_0", experts.tobytes()))
    tensors.append(("output.weight", (h, 4), "F32", np.zeros((4, h), np.float32).tobytes()))
    _write_meta(tmp_path / "mtp-m.gguf", {"general.architecture": "qwen4exp", "qwen4exp.block_count": 2,
                                          "qwen4exp.nextn_predict_layers": 1, "qwen4exp.embedding_length": h,
                                          "qwen4exp.hyper_connection.low_rank": rank}, tensors)
    assert has_mtp(tmp_path)
    ckpt = gguf.GGUFCheckpoint(tmp_path)
    assert "model.language_model.layers.0.linear_attn.norm.weight" in ckpt.names()
    got = {n: ckpt.tensor(n).numpy() for n in ckpt.names("mtp.")}
    np.testing.assert_array_equal(got["mtp.fc_embedding.weight"], eh[:, :h])
    np.testing.assert_array_equal(got["mtp.fc_hidden.weight"], eh[:, h:])
    np.testing.assert_array_equal(got["mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight"], inject[:rank])
    np.testing.assert_array_equal(got["mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"], inject[rank:])
    np.testing.assert_allclose(got["mtp.pre_fc_norm_embedding.weight"], enorm, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(got["mtp.layers.0.self_attn.q_proj.weight"], q)
    assert ckpt.shape("mtp.fc_hidden.weight") == (h, h)
    np.testing.assert_array_equal(ckpt.rows("mtp.fc_embedding.weight", 2, 5).numpy(), eh[2:5, :h])
    assert ckpt.experts("mtp.layers.0.mlp.experts")["gate"].type == "Q8_0"
    assert not any(n.startswith("model.") and "lm_head" in n for n in ckpt.names("mtp."))
