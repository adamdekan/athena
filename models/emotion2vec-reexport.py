#!/usr/bin/env python3
"""
emotion2vec-reexport.py — re-export with a DYNAMIC sequence axis, validated honestly.

WHY: the first export baked the audio alibi to the trace-time length (149 frames /
3 s). At runtime, every utterance of a different length dies on the /AUDIO/Reshape
(Input {16,149,149} vs requested {1,16,T,T}) — on BOTH CPU and CUDA EPs, because the
constant is in the .onnx itself. The old validation missed it: it only tried lengths
<= the trace length, and a baked-149 graph survives a 2 s clip but fails anything
longer. This script tries export strategies that keep the sequence dim SYMBOLIC, and
validates across SHORTER **and LONGER** clips (2,3,4,5,6 s -> 99,149,199,249,299
frames), asserting each one runs without a Reshape error AND matches FunASR.

Run offline, once:
    pip install -U "funasr" torch torchaudio modelscope onnx onnxruntime
    python3 emotion2vec-reexport.py 2>&1 | tee emotion2vec-reexport.out
Ship emotion2vec_plus_large.onnx only if a strategy reports FULLY DYNAMIC.
"""
import sys
import numpy as np
import torch

OUT = "emotion2vec_plus_large.onnx"
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral",
          "other", "sad", "surprised", "unk"]
TEST_SECS = [2, 3, 4, 5, 6]          # 3 s == the trace length; the rest must also pass
SR = 16000
SEP = "=" * 72


def n_frames(n):
    """wav2vec2/data2vec conv frontend output length: k[10,3,3,3,3,2,2] s[5,2,2,2,2,2,2]."""
    for k, s in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
        n = (n - k) // s + 1
    return n


class E2V(torch.nn.Module):
    """raw waveform [T] -> [1,9] softmax (same compute path that gave correct VALUES)."""
    def __init__(self, m, normalize):
        super().__init__()
        self.m = m
        self.normalize = bool(normalize)

    def forward(self, waveform):
        source = waveform
        if self.normalize:
            mean = source.mean()
            var = source.var(unbiased=False)
            source = (source - mean) / torch.sqrt(var + 1e-5)
        source = source.view(1, -1)
        if hasattr(self.m, "extract_features"):
            feats = self.m.extract_features(source, padding_mask=None)
        else:
            feats = self.m(source, padding_mask=None, mask=False,
                           features_only=True, remove_extra_tokens=True)
        x = feats["x"].mean(dim=1)
        x = self.m.proj(x)
        return torch.softmax(x, dim=-1)


def read_normalize(m):
    for g in (lambda: m.cfg.normalize, lambda: m.cfg.get("normalize"),
              lambda: getattr(m, "normalize", None)):
        try:
            v = g()
            if v is not None:
                return bool(v)
        except Exception:
            pass
    return True


def do_export(wrapper, sample, strategy):
    common = dict(input_names=["waveform"], output_names=["scores"], opset_version=17)
    if strategy == "legacy_nofold":
        torch.onnx.export(wrapper, (sample,), OUT,
                          dynamic_axes={"waveform": {0: "T"}},
                          do_constant_folding=False, dynamo=False, **common)
    elif strategy == "dynamo":
        # symbolic shapes via the FX/dynamo exporter (does not bake .size())
        try:
            T = torch.export.Dim("T")
        except Exception:
            from torch.export import Dim
            T = Dim("T")
        torch.onnx.export(wrapper, (sample,), OUT,
                          dynamo=True, dynamic_shapes={"waveform": {0: T}}, **common)
    else:
        raise ValueError(strategy)


def validate(am, so):
    import onnxruntime as ort  # noqa
    all_ok = True
    for secs in TEST_SECS:
        n = secs * SR
        wav = (0.05 * np.random.default_rng(secs).standard_normal(n)).astype(np.float32)
        ref = np.array([float(s) for s in am.generate(
            input=wav, granularity="utterance", extract_embedding=False)[0]["scores"]])
        try:
            onx = so.run(["scores"], {"waveform": wav})[0].reshape(-1)
            d = float(np.max(np.abs(onx - ref)))
            ok = d < 1e-3
            print(f"   {secs}s ({n_frames(n):3d} frames): "
                  f"{'OK' if ok else 'MISMATCH'}  max|onnx-FunASR|={d:.2e}  "
                  f"argmax={LABELS[int(np.argmax(onx))]}")
            all_ok &= ok
        except Exception as e:
            print(f"   {secs}s ({n_frames(n):3d} frames): RUN FAILED -> {repr(e)[:80]}")
            all_ok = False
    return all_ok


def main():
    print(SEP); print("torch", torch.__version__)
    from funasr import AutoModel
    am = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device="cpu",
                   disable_update=True, disable_pbar=True, log_level="ERROR")
    m = am.model.eval()
    normalize = read_normalize(m)
    print("cfg.normalize =", normalize)
    print("trace length 3s ->", n_frames(3 * SR), "frames (this is what got baked before)")
    wrapper = E2V(m, normalize).eval()
    sample = torch.from_numpy(
        (0.05 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32))

    import onnxruntime as ort
    for strategy in ("legacy_nofold", "dynamo"):
        print("\n" + SEP); print(f"strategy: {strategy}")
        try:
            do_export(wrapper, sample, strategy)
        except Exception as e:
            print(f"  export failed: {repr(e)[:300]}")
            continue
        try:
            so = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"  load failed: {repr(e)[:200]}")
            continue
        if validate(am, so):
            print(f"\n\u2713\u2713 '{strategy}' is FULLY DYNAMIC across {TEST_SECS}s. "
                  f"Ship {OUT}; re-enable emotion on the CUDA EP.")
            return
        print(f"  -> '{strategy}' still length-dependent (see failures above).")

    print("\n" + SEP)
    print("Neither strategy gave a fully dynamic model. Send emotion2vec-reexport.out;")
    print("next levers: fixed-length export + C++ padding, or patching the alibi op.")


if __name__ == "__main__":
    main()
