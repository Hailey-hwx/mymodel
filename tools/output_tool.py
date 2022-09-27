import imp
import json

# from accuracy_tool import gen_micro_macro_result

from .accuracy_tool import get_rouge


def null_output_function(data, config, *args, **params):
    return ""

# def basic_output_function(data, config, *args, **params):
#     which = config.get("output", "output_value").replace(" ", "").split(",")
#     temp = gen_micro_macro_result(data)
#     result = {}
#     for name in which:
#         result[name] = temp[name]

#     return json.dumps(result, sort_keys=True)

def rouge_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    # temp = get_rouge(data, config)
    temp = data
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)
