# llm_utils.py
# Helper to normalize LLM responses across different wrappers

from typing import Any

def normalize_llm_response(resp: Any) -> str:
    """
    Convert various LLM return types to a string safely.
    Handles:
      - plain str
      - objects with .content
      - objects with .text
      - objects with .generations (LangChain-like)
      - dict shapes (old OpenAI)
      - fallbacks to str(resp)
    """
    try:
        # plain string
        if isinstance(resp, str):
            return resp

        # LangChain v1-like: resp.content
        if hasattr(resp, "content"):
            c = getattr(resp, "content")
            if isinstance(c, str):
                return c
            # sometimes content can be a dict or list -> stringify
            return str(c)

        # object with text attribute
        if hasattr(resp, "text"):
            t = getattr(resp, "text")
            if isinstance(t, str):
                return t
            return str(t)

        # LangChain-like generations structure
        if hasattr(resp, "generations"):
            try:
                gen = getattr(resp, "generations")
                # try common nested list shape
                if isinstance(gen, (list, tuple)) and len(gen) > 0:
                    first = gen[0]
                    # first might be list as well
                    if isinstance(first, (list, tuple)) and len(first) > 0:
                        maybe = first[0]
                        if hasattr(maybe, "text"):
                            return getattr(maybe, "text")
                        if isinstance(maybe, dict) and "text" in maybe:
                            return maybe["text"]
                    # otherwise first might have .text
                    if hasattr(first, "text"):
                        return getattr(first, "text")
            except Exception:
                pass

        # older openai-style dict
        if isinstance(resp, dict):
            if "content" in resp and isinstance(resp["content"], str):
                return resp["content"]
            if "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if isinstance(ch, dict) and "text" in ch:
                    return ch["text"]
                if hasattr(ch, "text"):
                    return getattr(ch, "text")

        # fallback to str()
        return str(resp)
    except Exception:
        try:
            return str(resp)
        except Exception:
            return "<unserializable-llm-response>"

