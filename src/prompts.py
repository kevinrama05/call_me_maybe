"""Prompt construction for the Qwen3 chat template."""
from .function_definition_reader import FunctionDefinition


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def describe_function(function: FunctionDefinition) -> str:
    """Return a one-line description of a function for the system prompt.

    Args:
        function: The function to describe.

    Returns:
        A line such as ``- fn_add(a: number, b: number) -> number: Add.``
    """
    arguments = ", ".join(
        f"{name}: {param.type}" for name, param in function.parameters.items()
    )
    return (
        f"- {function.name}({arguments}) -> {function.returns.type}: "
        f"{function.description}"
    )


def build_system_prompt(functions: list[FunctionDefinition]) -> str:
    """Build the system message listing the available functions.

    Args:
        functions: Available functions.

    Returns:
        The system message text.
    """
    function_lines = "\n".join(describe_function(f) for f in functions)
    return (
        "You are a function-calling assistant. Choose the single function "
        "that best handles the user's request and extract its arguments "
        "from the request.\n"
        f"Available functions:\n{function_lines}\n\n"
        "Answer only with JSON of the form "
        '{"name":"<function>","parameters":{"<argument>":<value>}}.\n'
        "Numbers are written as JSON numbers. String arguments are copied "
        "exactly from the request, without the surrounding quotes, and "
        "escaped as JSON strings."
    )


def build_prompt(system: str, request: str) -> str:
    """Wrap a system message and a user request in the chat template.

    The assistant turn is opened with an empty thinking block so that the
    model directly starts answering.

    Args:
        system: System message.
        request: The user's natural-language request.

    Returns:
        The full prompt text, ready to be continued by the model.
    """
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{request}{IM_END}\n"
        f"{IM_START}assistant\n<think>\n\n</think>\n\n"
    )
