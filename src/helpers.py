"""Reading prompts and writing results."""
import json
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .function_definition_reader import InputError, read_json


class PromptInput(BaseModel):
    """One entry of the prompts input file."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def read_prompts(filepath: str) -> list[str]:
    """Load and validate the natural-language prompts.

    Args:
        filepath: Path of the prompts JSON file.

    Returns:
        The list of prompts, in file order.

    Raises:
        InputError: If the file is invalid.
    """
    data = read_json(filepath)
    if not isinstance(data, list):
        raise InputError(f"{filepath} must contain a JSON array")
    prompts: list[str] = []
    for index, item in enumerate(data):
        try:
            prompts.append(PromptInput.model_validate(item).prompt)
        except ValidationError:
            raise InputError(
                f"entry #{index} of {filepath} must be an object with a "
                f"string \"prompt\" field"
            )
    return prompts


def write_output(filepath: str, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON file, creating parent folders.

    Args:
        filepath: Destination path.
        calls: Function calls to write.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(
            [call.model_dump() for call in calls],
            file,
            indent=4,
            allow_nan=False,
        )
        file.write("\n")
