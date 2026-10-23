"""Loading and validation of the function definitions file."""
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


ParamType = Literal["string", "number", "integer", "boolean"]


class InputError(Exception):
    """Raised when an input file is missing, unreadable or malformed."""


class TypeDefinition(BaseModel):
    """Type of a parameter or of a return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """A callable function as described in functions_definition.json."""

    name: str = Field(min_length=1, pattern=r'^[^"\\\x00-\x1f]+$')
    description: str = ""
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


def read_json(filepath: str) -> Any:
    """Read and parse a JSON file.

    Args:
        filepath: Path of the file to read.

    Returns:
        The parsed JSON content.

    Raises:
        InputError: If the file cannot be read or is not valid JSON.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise InputError(f"{filepath} does not exist")
    except IsADirectoryError:
        raise InputError(f"{filepath} is a directory")
    except PermissionError:
        raise InputError(f"{filepath} cannot be read (permission denied)")
    except UnicodeDecodeError:
        raise InputError(f"{filepath} is not valid UTF-8")
    except json.JSONDecodeError as error:
        raise InputError(f"{filepath} has invalid JSON structure: {error}")
    except OSError as error:
        raise InputError(f"{filepath} cannot be read: {error}")


def functions_definition_validator(filepath: str) -> list[FunctionDefinition]:
    """Load and validate every function of a definitions file.

    Args:
        filepath: Path of the functions definition JSON file.

    Returns:
        The validated function definitions.

    Raises:
        InputError: If the file is invalid or contains no usable function.
    """
    raw_json = read_json(filepath)
    if not isinstance(raw_json, list):
        raise InputError(f"{filepath} must contain a JSON array")
    functions: list[FunctionDefinition] = []
    for index, item in enumerate(raw_json):
        try:
            functions.append(FunctionDefinition.model_validate(item))
        except ValidationError as error:
            raise InputError(
                f"invalid function definition #{index} in {filepath}:\n"
                f"{error}"
            )
    if not functions:
        raise InputError(f"{filepath} does not define any function")
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise InputError(f"{filepath} contains duplicate function names")
    return functions
