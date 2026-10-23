"""Command-line entry point: ``uv run python -m src``."""
import argparse
import sys
import time

from .function_definition_reader import (
    InputError,
    functions_definition_validator,
)
from .helpers import FunctionCall, read_prompts, write_output


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="functions definition file",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="prompts file",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="output file",
    )
    return parser.parse_args()


def main() -> int:
    """Run the whole pipeline.

    Returns:
        The process exit status.
    """
    args = parse_arguments()
    try:
        functions = functions_definition_validator(args.functions_definition)
        prompts = read_prompts(args.input)
    except InputError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    print("Loading the model...", file=sys.stderr)
    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        from .LLMAnswers import LLMAnswers

        llm = LLMAnswers(functions=functions, sdk=Small_LLM_Model())
    except Exception as error:
        print(f"[ERROR] cannot load the model: {error}", file=sys.stderr)
        return 1

    start = time.monotonic()
    calls: list[FunctionCall] = []
    for index, prompt in enumerate(prompts, 1):
        try:
            call = llm.call_function(prompt)
        except Exception as error:
            print(
                f"[ERROR] prompt {index}/{len(prompts)} {prompt!r}: {error}",
                file=sys.stderr,
            )
            continue
        calls.append(call)
        print(
            f"[{index}/{len(prompts)}] {prompt!r} -> "
            f"{call.name}({call.parameters})",
            file=sys.stderr,
        )
    print(
        f"Processed {len(calls)}/{len(prompts)} prompts in "
        f"{time.monotonic() - start:.1f}s",
        file=sys.stderr,
    )

    try:
        write_output(args.output, calls)
    except (OSError, ValueError) as error:
        print(f"[ERROR] cannot write {args.output}: {error}", file=sys.stderr)
        return 1
    print(f"Results written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
