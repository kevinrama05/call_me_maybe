"""Constrained generation of function calls with the LLM."""
import json
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .constraints import SCANNERS, Scanner, Status, make_choice_scanner
from .function_definition_reader import FunctionDefinition
from .helpers import FunctionCall
from .prompts import build_prompt, build_system_prompt


MAX_STEPS = 600


def bytes_to_unicode() -> dict[int, str]:
    """Return the byte-level BPE mapping from bytes to printable characters.

    Byte-level BPE vocabularies (GPT-2, Qwen...) store every token as a
    string where each byte is replaced by a printable character, e.g. a
    space becomes ``Ġ``. This rebuilds that table.

    Returns:
        A mapping from each byte value to its printable character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, characters)}


def load_token_bytes(vocab_path: str) -> list[bytes]:
    """Map every token id of the vocabulary file to its raw bytes.

    Args:
        vocab_path: Path of the ``vocab.json`` file of the tokenizer.

    Returns:
        A list indexed by token id. Ids missing from the file are empty.
    """
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab: dict[str, int] = json.load(file)
    char_to_byte = {char: byte for byte, char in bytes_to_unicode().items()}
    token_bytes: list[bytes] = [b""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        try:
            token_bytes[token_id] = bytes(char_to_byte[c] for c in token)
        except KeyError:
            continue
    return token_bytes


class LLMAnswers(BaseModel):
    """Translate prompts into function calls with constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: list[FunctionDefinition]
    sdk: Any
    token_bytes: list[bytes] = Field(default_factory=list)
    system_prompt: str = ""

    def model_post_init(self, context: Any) -> None:
        """Load the vocabulary and prepare the system prompt."""
        self.token_bytes = load_token_bytes(self.sdk.get_path_to_vocab_file())
        self.system_prompt = build_system_prompt(self.functions)

    def encode(self, text: str) -> list[int]:
        """Tokenize a text into a list of token ids."""
        ids: list[int] = self.sdk.encode(text)[0].tolist()
        return ids

    def generate_value(self, context: list[int], scanner: Scanner) -> bytes:
        """Generate one JSON value under a grammar constraint.

        At each step the logits of every token that would make the value
        invalid are discarded (masked to -inf), and the best remaining
        token is chosen. Rather than building the full mask, tokens are
        examined from the most to the least likely and the first valid
        one is kept, which selects exactly the same token.

        Args:
            context: Token ids of the prompt and JSON written so far.
            scanner: Grammar check of the value being generated.

        Returns:
            The raw bytes of the generated value.

        Raises:
            RuntimeError: If no token can continue the value.
        """
        context = list(context)
        buffer = b""
        for _ in range(MAX_STEPS):
            logits = np.asarray(
                self.sdk.get_logits_from_input_ids(context), dtype=np.float32
            )
            vocab_size = min(len(logits), len(self.token_bytes))
            for token_id in np.argsort(-logits[:vocab_size]):
                token = self.token_bytes[token_id]
                if not token:
                    continue
                status, value = scanner(buffer + token)
                if status is not Status.INVALID:
                    break
            else:
                raise RuntimeError("no valid token can continue the output")
            if status is Status.DONE:
                return value
            buffer += token
            context.append(int(token_id))
        raise RuntimeError("generation did not finish")

    def call_function(self, request: str) -> FunctionCall:
        """Produce the function call answering one request.

        The JSON structure (braces, keys, quotes, commas) is written by the
        program; the model only fills in the function name and the
        argument values, each one under its own type constraint.

        Args:
            request: The natural-language request.

        Returns:
            The validated function call.
        """
        prompt = build_prompt(self.system_prompt, request)
        names = [function.name.encode() for function in self.functions]
        partial = '{"name":"'
        name_scanner = make_choice_scanner(names, quoted=True)
        name = self.generate_value(
            self.encode(prompt + partial), name_scanner
        ).decode()
        function = next(f for f in self.functions if f.name == name)

        parameters: dict[str, Any] = {}
        partial += json.dumps(name)[1:] + ',"parameters":{'
        for index, (key, definition) in enumerate(
            function.parameters.items()
        ):
            if index:
                partial += ","
            partial += json.dumps(key) + ":"
            if definition.type == "string":
                partial += '"'
            raw = self.generate_value(
                self.encode(prompt + partial), SCANNERS[definition.type]
            )
            text = raw.decode("utf-8", errors="replace")
            if definition.type == "string":
                value: Any = json.loads(f'"{text}"')
                partial += f'{text}"'
            else:
                value = json.loads(text)
                if definition.type == "number":
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"{text} is out of range")
                partial += text
            parameters[key] = value
        return FunctionCall(prompt=request, name=name, parameters=parameters)
