from dualstream.integrity import RunningHash, verify_running_hash
from dualstream.frame import MonologueFrameV1, TopKToken, encode_frame

def test_running_hash():
    h = RunningHash()
    frames = []
    for i in range(3):
        fr = MonologueFrameV1(prompt_nonce=1, token_index=i, chosen_id=i, topk=[TopKToken(i, 1.0)])
        fb = encode_frame(fr, include_crc32=True)
        frames.append(fb)
        h.update(fb)
    assert verify_running_hash(frames, h.digest_hex())
