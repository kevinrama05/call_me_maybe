# from llm_sdk import Small_LLM_Model
import json
from sys import argv
from function_definition_reader import read_json, functions_list

functions = read_json(argv[1])
function_list_ = functions_list(functions)


function_definition_helper = (
        "You are a function-calling assistant. Given a user request, "
        "respond with the name of exactly one function from this list "
        "that best handles it:\n" + function_list_
    )

parameter_helper = (
        f"The user's request is: \"{user_question}\"\n"
        f"You've determined this calls the function `{fn_def.name}` "
        f"({fn_def.description}), which takes these parameters:\n{param_lines}\n"
        "Extract the correct value for each parameter from the user's request."
    )

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

question = "What will the weather be tomorrow, if today is June 7?"
prompt = f"{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n<think>\n\n</think>\n\n"

sdk = Small_LLM_Model()

token_ids = sdk._encode(prompt)
token_ids = token_ids[0].tolist()

stop_ids_tensor = sdk._encode(IM_END)
stop_ids = stop_ids_tensor[0].tolist()
stop_id = stop_ids[-1]

answer = []
for i in range(200):
    logits = sdk.get_logits_from_input_ids(token_ids)
    next_token = max(range(len(logits)), key=lambda idx: logits[idx])

    if next_token == stop_id:
        break

    answer.append(next_token)
    token_ids.append(next_token)

print(sdk._decode(answer))