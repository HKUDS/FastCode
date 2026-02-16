from openai import BadRequestError


def openai_chat_completion(client, *, max_tokens, **kwargs):
    """Call OpenAI-compatible chat completions with max_tokens fallback.

    Tries max_tokens first (broadest compatibility), falls back to
    max_completion_tokens if the model rejects max_tokens (e.g. gpt-5.2, o1).
    """
    try:
        return client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    except BadRequestError as e:
        if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
            return client.chat.completions.create(
                max_completion_tokens=max_tokens, **kwargs
            )
        raise
