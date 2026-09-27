from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceguard.config import load_config
from voiceguard.d2.analysis import analyze
from voiceguard.d2.predictors import dissertation_verdict
from voiceguard.d2.reference import Reference, build
from voiceguard.dsp.vad import AudioClip, segment


def prng_check_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PRNG Dissertation Verification Mode")
    parser.add_argument("--gen", choices=["lfsr", "mt", "urandom"], default="urandom")
    parser.add_argument("--n", type=int, default=100000)
    args = parser.parse_args(argv)

    try:
        from tests.prng_fixtures import generate_lfsr, generate_mt, generate_urandom

        if args.gen == "lfsr":
            bits = generate_lfsr(args.n)
        elif args.gen == "mt":
            bits = generate_mt(args.n)
        else:
            bits = generate_urandom(args.n)
    except ImportError:
        # Fallback inline generators
        if args.gen == "lfsr":
            bits = np.zeros(args.n, dtype=np.uint8)
            state = 0xACE12345
            for i in range(args.n):
                bits[i] = state & 1
                fb = ((state >> 0) ^ (state >> 1) ^ (state >> 21) ^ (state >> 31)) & 1
                state = (state >> 1) | (fb << 31)
        elif args.gen == "mt":
            import random

            rng = random.Random(42)
            words = [rng.getrandbits(32) for _ in range((args.n + 31) // 32)]
            bits = np.unpackbits(np.array(words, dtype=np.uint32).view(np.uint8))[: args.n]
        else:
            import os

            raw = os.urandom((args.n + 7) // 8)
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: args.n]

    cfg = load_config()
    res = dissertation_verdict(bits, cfg=cfg)

    print(f"Generator: {args.gen} (N={args.n} bits)")
    print(f"delta_max: {res['delta_max']:.5f}")
    print(f"threshold eps: {res['eps']:.5f}")
    print(f"verdict: {res['verdict']} (is_violation={res['is_violation']})")
    return 0


def build_ref_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build Live Speech Reference Model")
    parser.add_argument("--channel", default="amrnb_12.2")
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--frames", default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    out_path = args.out
    if out_path is None:
        out_path = (
            Path(cfg.paths.models)
            / "reference"
            / f"{args.variant}_{args.frames}_{args.channel}.json"
        )

    ref = build(channel=args.channel, variant=args.variant, frames=args.frames, cfg=cfg)
    ref.save(out_path)
    print(f"Reference saved to: {out_path}")
    return 0


def analyze_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Analyze Speech Clip with D2 Detector")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--channel", default="clean")
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--frames", default="all")
    parser.add_argument("--audio", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    ref_path = (
        Path(cfg.paths.models) / "reference" / f"{args.variant}_{args.frames}_{args.channel}.json"
    )

    if ref_path.is_file():
        ref = Reference.load(ref_path, cfg=cfg)
    else:
        # Build fallback reference on the fly
        ref = build(channel=args.channel, variant=args.variant, frames=args.frames, cfg=cfg)

    if args.audio and Path(args.audio).is_file():
        samples, sr = sf.read(args.audio, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
    else:
        # Synthetic fixture
        sr = 8000
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        samples = (0.6 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    clip = AudioClip(samples=samples, sr=sr, clip_id=args.clip_id)
    segments = segment(clip, cfg=cfg)

    print(f"Clip '{args.clip_id}': {len(segments)} segments analyzed (Channel: {args.channel})")
    for seg in segments:
        result = analyze(seg, ref=ref, cfg=cfg)
        print(f"--- Segment {seg.index} ---")
        print(f"  s2: {result.s2:.4f}, n_symbols: {result.n_symbols}")
        print(f"  chi2_p: {result.chi2_p}")
        for cid in sorted(result.delta.keys()):
            d_val = result.delta[cid]
            z_val = result.z.get(cid, float("nan"))
            print(f"    {cid}: delta={d_val:.4f}, Z={z_val:.2f}")

    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m voiceguard.d2 {prng-check,build-ref,analyze} [options]")
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "prng-check":
        return prng_check_main(rest)
    elif cmd == "build-ref":
        return build_ref_main(rest)
    elif cmd == "analyze":
        return analyze_main(rest)
    else:
        print(f"Unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
