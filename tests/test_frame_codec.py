import random
from dualstream.frame import MonologueFrameV1, TopKToken, AttnSummary, ConceptScore, encode_frame, decode_frame

def test_encode_decode_roundtrip_with_crc():
    fr = MonologueFrameV1(
        prompt_nonce=123,
        token_index=7,
        chosen_id=42,
        topk=[TopKToken(1, 0.5), TopKToken(2, 0.25)],
        attn=[AttnSummary(layer=0, head=1, top_token_idx=3, weight=0.9)],
        concepts=[ConceptScore(concept_id=1001, score=0.83)],
    )
    b = encode_frame(fr, include_crc32=True)
    fr2 = decode_frame(b, require_crc32=True)
    assert fr2.prompt_nonce == 123
    assert fr2.token_index == 7
    assert fr2.chosen_id == 42
    assert len(fr2.topk) == 2
    assert abs(fr2.topk[0].prob - 0.5) < 1e-2  # float16 rounding
    assert fr2.crc32 is not None

def test_decode_without_crc():
    fr = MonologueFrameV1(
        prompt_nonce=999,
        token_index=0,
        chosen_id=5,
        topk=[TopKToken(5, 1.0)],
    )
    b = encode_frame(fr, include_crc32=False)
    fr2 = decode_frame(b, require_crc32=False)
    assert fr2.crc32 is None
