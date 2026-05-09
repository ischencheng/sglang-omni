# SPDX-License-Identifier: Apache-2.0
"""Reference script: launch Qwen3-Omni speech with SGLang-backed encoder TP.

This is the Phase-0 deployment recipe described in
``docs/developer_reference/encoder_tp_path_b_design.md`` (RFC #375).
It shows how to flip the image and audio encoder stages from the local
HF tower to the SGLang-native encoders without changing any other
stage.

Usage (single-host, 8-GPU layout):

    python examples/qwen3_omni_encoder_tp.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --image-tp 2 --audio-tp 2 --port 8000

The encoders run with ``tp_size=2`` each on dedicated GPU pairs; the
thinker, talker, and code2wav stages keep their existing single-GPU
placement. ``backend="auto"`` would resolve to ``"local"`` in Phase 0
(see RFC Phase 2 for the flip); this script picks ``"sglang"``
explicitly so the SGLang encoder worker takes over right now.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--image-tp", type=int, default=2)
    p.add_argument("--audio-tp", type=int, default=2)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    # Import after argparse so --help is fast.
    from sglang_omni_v1.models.qwen3_omni.config import (
        Qwen3OmniSpeechPipelineConfig,
    )
    from sglang_omni_v1.serve.launcher import launch_server

    if args.image_tp + args.audio_tp + 4 > 8:
        raise SystemExit(
            "This recipe assumes 8 GPUs: image_encoder TP + audio_encoder TP "
            "+ thinker + talker + code2wav + decode/aggregate scratch."
        )

    image_gpus = list(range(0, args.image_tp))
    audio_gpus = list(range(args.image_tp, args.image_tp + args.audio_tp))

    # Build the canonical config and flip just the two encoder stages.
    cfg = Qwen3OmniSpeechPipelineConfig(model_path=args.model)
    new_stages = []
    for s in cfg.stages:
        if s.name == "image_encoder":
            s = s.model_copy(update={
                "factory_args": {**s.factory_args, "backend": "sglang"},
                "tp_size": args.image_tp,
                "gpu": image_gpus,
            })
        elif s.name == "audio_encoder":
            s = s.model_copy(update={
                "factory_args": {**s.factory_args, "backend": "sglang"},
                "tp_size": args.audio_tp,
                "gpu": audio_gpus,
            })
        new_stages.append(s)
    cfg = cfg.model_copy(update={"stages": new_stages})

    launch_server(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
