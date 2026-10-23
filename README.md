*This activity has been created as part of the 42 curriculum by kerama.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system powered by a Small Language Model (LLM).

The objective of this project is to convert natural language requests into structured function calls.
Instead of answering the user's question directly, the program determines which function should be called and extracts the required parameters.

Example:

Input:

```text
What is the sum of 2 and 3?
```

Output:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2,
        "b": 3
    }
}
```

The project uses constrained decoding to guarantee that every generated output is valid JSON and respects the provided function schema.

---

# Features

- Reads function definitions from JSON.
- Reads prompts from JSON.
- Uses Qwen3-0.6B through the provided SDK.
- Performs constrained decoding.
- Produces schema-valid JSON.
- Gracefully handles invalid files and malformed JSON.
- Supports custom input/output paths.

---

# Instructions

## Requirements

- Python 3.10+
- uv
- numpy
- pydantic

## Installation

```bash
make install
```

or

```bash
uv sync
```

---

## Running

```bash
make run
```

or

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

---

## Debugging

```bash
make debug
```

---

## Lint

```bash
make lint
```

---

## Cleaning

```bash
make clean
```

---

# Algorithm Explanation

The implementation processes each prompt in two consecutive LLM inference stages.

### 1. Function Selection

For every natural language prompt, the program first builds a helper prompt containing the list of available functions extracted from `functions_definition.json`.

The prompt is then sent to the LLM using the provided SDK. Generation is performed greedily by repeatedly:

1. Encoding the prompt into token IDs.
2. Obtaining the logits for the next token.
3. Selecting the token with the highest probability (`argmax`).
4. Appending the selected token to the input sequence.
5. Repeating until the model generates the `<|im_end|>` stop token.

The generated text is interpreted as the name of the function that best matches the user's request.

### 2. Parameter Generation

After selecting the function, the corresponding function definition is retrieved.

A second prompt is created containing:

- the selected function definition,
- the original user prompt,
- instructions describing the expected JSON structure.

The LLM is queried again using the same greedy decoding process. This second generation produces a JSON object containing the selected function name and its parameters.

### 3. Validation

The generated JSON is processed before validation:

- the original prompt is inserted into the JSON object;
- the JSON string is parsed using Python's `json` module;
- the result is validated with a Pydantic model (`ValidatePrompt`).

Pydantic verifies that every generated object contains the required fields:

- `prompt`
- `name`
- `parameters`

If validation fails or the generated JSON is malformed, an exception is raised, preventing invalid output from being written.

### 4. Output Generation

Each validated function call is stored as a `ValidatePrompt` object.

After all prompts have been processed, the objects are converted back into dictionaries and written to the output file as a formatted JSON array.

---

# Design Decisions

## Pydantic

Pydantic is used for validating:

- input files
- function definitions
- output objects

This guarantees that every produced function call follows the required schema.

## State Machine

Instead of relying on prompting, the decoder uses a finite-state machine that knows exactly which tokens are allowed at every stage.

## Vocabulary

The SDK vocabulary file is loaded once during startup to efficiently translate token IDs into strings.

## Error Handling

The application never crashes unexpectedly.

Errors handled include:

- missing files
- invalid JSON
- invalid schemas
- unsupported functions
- malformed prompts

---

# Performance Analysis

The target objectives are:

- 90%+ function selection accuracy
- 100% valid JSON
- execution under five minutes

Most computation time is spent obtaining logits from the model.

Vocabulary loading is performed once to minimize overhead.

---

# Challenges Faced

The main challenges were:

- understanding tokenizer behavior
- implementing constrained decoding
- handling partial JSON generation
- restricting numbers, strings and booleans correctly
- maintaining JSON validity after every generated token

The solution uses a deterministic parser state to guarantee correctness.

---

# Testing Strategy

The project was tested with:

- empty prompts
- long prompts
- large numbers
- floating-point numbers
- booleans
- invalid JSON files
- missing files
- malformed schemas
- unknown functions

The generated JSON was validated using both Python's `json` module and Pydantic models.

---

# Example Usage

Input:

```json
[
    {
        "prompt": "What is the sum of 40 and 2?"
    }
]
```

Output:

```json
[
    {
        "prompt": "What is the sum of 40 and 2?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 40,
            "b": 2
        }
    }
]
```

---

# Project Structure

```
.
├── data/
│   ├── input/
│   └── output/
├── llm_sdk/
├── src/
├── pyproject.toml
├── uv.lock
├── README.md
└── Makefile
```

---

# Resources

## Documentation

- Python Documentation
- Pydantic Documentation
- NumPy Documentation

## LLM References

- Qwen3 Technical Report
- JSON Specification (RFC 8259)
- Finite State Machines
- Constrained Decoding research papers

## AI Usage

AI was used for:

- discussing constrained decoding ideas
- understanding tokenizer behavior
- reviewing implementation concepts
- improving documentation

All implementation decisions and source code were reviewed, understood, and validated manually.