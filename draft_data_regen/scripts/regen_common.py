"""Request and validation helpers shared by the regeneration scripts.

Vendored from SpecForge (MIT, https://github.com/sgl-project/SpecForge):
scripts/conversation_validation.py and the parts of scripts/regenerate_train_data.py that
regen_driver.py uses, so this directory runs without a SpecForge checkout. Same request
format and row schema as SpecForge's regenerator.
"""

from typing import Any, Dict

from openai import OpenAI  # noqa: F401  (regen_driver.py wraps it with a long timeout)


def has_think_marker(content: str) -> bool:
    lowered = content.lower()
    return "<think>" in lowered or "</think>" in lowered


def validate_conversation(
    messages: Any, *, error_style: str = "validation"
) -> str | None:
    """Why a conversation is not optional-system + alternating user/assistant turns with
    non-empty string content, or None."""
    regen = error_style == "regeneration"
    if not isinstance(messages, list) or not messages:
        return (
            "Missing or empty conversations list"
            if regen
            else "conversations must be a non-empty list"
        )
    expected_role = "user"
    saw_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return (
                f"Invalid message at position {index}: expected object"
                if regen
                else f"message {index} must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return (
                f"Invalid message content at position {index}: expected non-empty string"
                if regen
                else f"message {index} content must be a non-empty string"
            )
        if role == "system" and not saw_user:
            continue
        if role not in {"user", "assistant"}:
            return (
                f"Invalid message role: {role}"
                if regen
                else f"message {index} has unsupported role {role!r}"
            )
        if role != expected_role:
            if regen:
                if not saw_user and role == "assistant":
                    return "Data starts with an assistant message"
                return f"Invalid conversation role order at position {index}: expected {expected_role}, got {role}"
            return f"message {index} expected role {expected_role!r}, got {role!r}"
        saw_user = True
        expected_role = "assistant" if role == "user" else "user"
    if not saw_user:
        return (
            "Data does not contain a user message"
            if regen
            else "conversation has no user message"
        )
    return None


def validate_regen_input(data: Any) -> str | None:
    """Why a row cannot be regenerated, or None."""
    if not isinstance(data, dict):
        return "Expected a JSON object"
    return validate_conversation(data.get("conversations"), error_style="regeneration")


def set_skipped(data: Any, error: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "skipped", "error": error, "data": data}
    data["status"] = "skipped"
    data["error"] = error
    return data


def build_query_kwargs(args, messages, max_tokens=None):
    """OpenAI chat-completion kwargs for one turn. With args.reasoning == "save" earlier
    answers are sent without their reasoning_content and thinking is enabled; "disable"
    turns thinking off through chat_template_kwargs."""
    query_messages = messages
    if args.reasoning == "save":
        query_messages = []
        for message in messages:
            query_message = dict(message)
            if query_message.get("role") == "assistant":
                query_message.pop("reasoning_content", None)
            query_messages.append(query_message)
    query_kwargs = dict(
        model=args.model,
        messages=query_messages,
        max_tokens=max_tokens if max_tokens is not None else args.max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.reasoning == "disable":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.reasoning == "save":
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    return query_kwargs
