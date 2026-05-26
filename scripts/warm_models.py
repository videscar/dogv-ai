from __future__ import annotations

from datetime import datetime, timezone
import logging
import time

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import get_settings
from api.embed import EmbedClient
from api.llm import LlmClient


logger = logging.getLogger("dogv.warm_models")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    llm = LlmClient()
    embedder = EmbedClient()

    logger.info(
        "warmup.start llm=%s@%s embed=%s@%s",
        settings.llm_model,
        settings.llm_base_url,
        settings.embed_model,
        settings.embed_base_url,
    )

    started = time.monotonic()
    try:
        embed_start = time.monotonic()
        embedding = embedder.embed("warmup embedding request")
        if not embedding:
            raise RuntimeError("Empty embedding returned by embed server")
        logger.info(
            "warmup.embed.ok dims=%s elapsed=%.2fs",
            len(embedding),
            time.monotonic() - embed_start,
        )

        chat_start = time.monotonic()
        text = llm.chat(
            [
                {
                    "role": "user",
                    "content": "Responde unicamente con la palabra OK.",
                }
            ],
            temperature=0.0,
        )
        if not text.strip():
            raise RuntimeError("Empty chat response returned by LLM server")
        logger.info(
            "warmup.chat.ok chars=%s elapsed=%.2fs",
            len(text),
            time.monotonic() - chat_start,
        )

        logger.info(
            "warmup.done status=ok total_elapsed=%.2fs at=%s",
            time.monotonic() - started,
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        return 0
    except Exception:
        logger.exception("warmup.done status=failed elapsed=%.2fs", time.monotonic() - started)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
