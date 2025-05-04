#!/usr/bin/env python3
"""
ld_pumper_stream.py  –  Bedrock + LaunchDarkly hammer that
• measures time-to-first-token
• reports full usage / cost metrics
• records Success + probabilistic Satisfaction based on seed
"""

import os, sys, time, random, signal, logging, pathlib
from typing import Any, Dict, List
from dotenv import load_dotenv

# ────── creds from .env ────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

for k in ("LD_SERVER_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    if not os.getenv(k):
        sys.exit(f"Missing `{k}` in .env")

# ────── SDKs ───────────────────────────────────────────────────────
import ldclient
from ldclient.context import Context
from ldclient.config  import Config
from ldai.client      import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig
from ldai.tracker     import FeedbackKind

import boto3
from botocore.exceptions import ClientError

# ────── logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ld_pumper_stream")

# ────── wrappers ───────────────────────────────────────────────────
class LaunchDarklyClient:
    def __init__(self, sdk_key: str, ai_cfg: str):
        ldclient.set_config(Config(sdk_key))
        self.ld  = ldclient.get()
        self.ai  = LDAIClient(self.ld)
        self.cfg = ai_cfg

    def get_config(self, ctx: Context, variables: Dict[str, Any]):
        fallback = AIConfig(
            enabled=True,
            model=ModelConfig(
                name="anthropic.claude-v2:1",
                parameters={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000},
            ),
            messages=[LDMessage(role="system", content="You are a helpful insurance assistant.")],
            provider=ProviderConfig(name="bedrock"),
        )
        return self.ai.config(self.cfg, ctx, fallback, variables)

class BedrockClient:
    def __init__(self, region: str):
        self.rt = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def converse_stream(self, **params):
        return self.rt.converse_stream(**params)["stream"]

# ────── helpers ────────────────────────────────────────────────────
PROMPT, PAUSE = "provide me information about my claim", 1.0

def build_messages(prompt: str):
    return [{"role": "user", "content": [{"text": prompt}]}]

def shutdown(sig, _):
    log.warning("Received %s – exiting", sig.name)
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

def positive_feedback(seed: int) -> bool:
    """95 % positive if seed ≥ 5; 85 % positive otherwise."""
    threshold = 0.95 if seed >= 5 else 0.85
    return random.random() < threshold

# ────── main loop ──────────────────────────────────────────────────
def main():
    ldc = LaunchDarklyClient(os.getenv("LD_SERVER_KEY"),
                             os.getenv("LD_AI_CONFIG_ID", "test1"))
    bed = BedrockClient(os.getenv("AWS_REGION", "us-east-1"))

    while True:
        try:
            seed = random.randint(1, 10)
            ctx  = Context.builder("demo-user").set("seed", seed).build()
            ldc.ld.identify(ctx)

            cfg, tracker = ldc.get_config(ctx, {"user_input": PROMPT})
            sys_prompt   = [{"text": cfg.messages[0].content}]
            p            = cfg.model._parameters or {}
            infer_cfg    = {
                "temperature": p.get("temperature"),
                "topP":        p.get("top_p"),
                "maxTokens":   p.get("max_tokens"),
            }
            infer_cfg = {k: v for k, v in infer_cfg.items() if v is not None}

            # ── 1) request + timing ───────────────────────────
            t0 = time.perf_counter()
            stream = bed.converse_stream(
                modelId=cfg.model.name,
                messages=build_messages(PROMPT),
                system=sys_prompt,
                inferenceConfig=infer_cfg,
                additionalModelRequestFields={},
            )

            full_text, first_ms = "", None
            bed_resp = {"usage": {}, "metrics": {}, "$metadata": {}}

            for ev in stream:
                if "contentBlockDelta" in ev:
                    if first_ms is None:
                        first_ms = (time.perf_counter() - t0) * 1000
                    full_text += ev["contentBlockDelta"]["delta"]["text"]

                if "metadata" in ev:
                    md = ev["metadata"]
                    bed_resp["metadata"] = md
                    if "usage"   in md: bed_resp["usage"]   = md["usage"]
                    if "metrics" in md: bed_resp["metrics"] = md["metrics"]

            log.info("seed=%d | ttf=%.0f ms | pos? %s | %s…",
                     seed, first_ms or -1,
                     "yes" if positive_feedback(seed) else "no",
                     full_text[:70].replace("\n", " "))

            # ── 2) LD metrics + probabilistic feedback ───────
            if tracker:
                tracker.track_bedrock_converse_metrics(bed_resp)
                if first_ms is not None:
                    tracker.track_time_to_first_token(first_ms)
                tracker.track_success()

                is_positive = positive_feedback(seed)
                tracker.track_feedback({"kind":
                                         FeedbackKind.Positive if is_positive
                                         else FeedbackKind.Negative})
                ldc.ld.flush()

        except ClientError as e:
            log.error("Bedrock error: %s", e)
        except Exception as e:
            log.exception("loop error: %s", e)

        time.sleep(PAUSE)

# ────── run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
