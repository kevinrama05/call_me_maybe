import json

def read_json(filepath: str) -> list[dict[str, str]]:
    with open(filepath, "r") as file:
        functions = json.load(file)

    return functions

def functions_list(json_text: list[dict[str, str]]) -> str:
    function_list = "\n"
    for i in json_text:
        for key, value in i.items():
            if key == 'fn_name':
                function_list += f"- {value}\n"
    
    return function_list