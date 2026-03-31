import types

import torch

from dualstream.generator import DualStreamGenerator, GenerationConfig


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None
    bos_token_id = None
    additional_special_tokens_ids = None

    def __call__(self, _prompt, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens=False):
        if isinstance(tokens, list) and tokens and tokens[0] == 0:
            return "<eos>"
        return "tok"


class _FakeModel:
    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states,
    ):
        logits = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=torch.float32)
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=None,
            attentions=None,
            hidden_states=None,
        )


def test_probe_pack_path_not_touched_when_probes_disabled(monkeypatch):
    def _forbidden_loader(_path):
        raise AssertionError("Probe pack loader must not be called when probes are disabled")

    monkeypatch.setattr("dualstream.generator.ProbePack.from_json", _forbidden_loader)

    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name = "fake"
    gen.device = "cpu"
    gen.tokenizer = _FakeTokenizer()
    gen.model = _FakeModel()

    cfg = GenerationConfig(
        include_probes=False,
        probe_pack_path="/definitely/missing/probe_pack.json",
        max_new_tokens=1,
        top_k=1,
        do_sample=False,
        include_running_hash=False,
        include_crc32=False,
        enable_heuristics=False,
    )

    result = gen.generate("test", cfg)
    assert result["frames"][0].probe_pack_id is None
    assert result["frames"][0].probe_pack_hash is None
