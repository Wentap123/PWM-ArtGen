system_prompt = """
You are an expert in the recognition of articulated parts of an object in an image. 
The categories of objects include: storage furniture, table, refrigerator, oven, microwave, washer, dishwasher.
You will be provided with an image of an articulated object.


You should follow the following steps to achieve the task:
1) Recognize all the articulated parts of the object in the image, choosing only from ['base', 'door', 'drawer']. There must always be exactly one "base".
2) Describe how the parts are connected and then organize them in a part connectivity graph. The "base" part is always the root of the graph. 
3) All 'door' and 'drawer' parts attach directly to 'base'. Do not include any other part types.

Here is an example of your response:

I recognize all the articulated parts of a storage furniture, they are: base, door (attach to base), drawer (attach to base).
The part connectivity graph for the object is:
```json
{"base": [{"door": []}, {"drawer": []}]}
```
"""

example_prompt_1 = "An image of a storage furniture that has four doors."
example_assistant_1 = (
    "I recognize all the articulated parts in a storage furniture, they are: base, "
    "door (attach to base), door (attach to base), door (attach to base), door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}, {\"door\": []}, {\"door\": []}, {\"door\": []}]}\n"
    "```\n"
)
example_prompt_2 = "An image of a table that has two drawers."
example_assistant_2 = (
    "I recognize all the articulated parts in a table, they are: base, "
    "drawer (attach to base), drawer (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"drawer\": []}, {\"drawer\": []}]}\n"
    "```\n"
)
example_prompt_3 = "An image of a refrigerator that has two doors."
example_assistant_3 = (
    "I recognize all the articulated parts in a refrigerator, they are: base, "
    "door (attach to base), door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}, {\"door\": []}]}\n"
    "```\n"
)

example_prompt_4 = "An image of an oven that has two doors."
example_assistant_4 = (
    "I recognize all the articulated parts in an oven, they are: base, "
    "door (attach to base), door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}, {\"door\": []}]}\n"
    "```\n"
)

example_prompt_5 = "An image of a microwave that has one door."
example_assistant_5 = (
    "I recognize all the articulated parts in a microwave, they are: base, "
    "door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}]}\n"
    "```\n"
)

example_prompt_6 = "An image of a washer that has one door."
example_assistant_6 = (
    "I recognize all the articulated parts in a washer, they are: base, "
    "door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}]}\n"
    "```\n"
)

example_prompt_7 = "An image of a dishwasher that has one door."
example_assistant_7 = (
    "I recognize all the articulated parts in a dishwasher, they are: base, "
    "door (attach to base).\n\n"
    "The part connectivity graph for the object is:\n"
    "```json\n"
    "{\"base\": [{\"door\": []}]}\n"
    "```\n"
)

examples = [
    {'prompt': example_prompt_1, 'assistant': example_assistant_1},
    {'prompt': example_prompt_2, 'assistant': example_assistant_2},
    {'prompt': example_prompt_3, 'assistant': example_assistant_3},
    {'prompt': example_prompt_4, 'assistant': example_assistant_4},
    {'prompt': example_prompt_5, 'assistant': example_assistant_5},
    {'prompt': example_prompt_6, 'assistant': example_assistant_6},
    {'prompt': example_prompt_7, 'assistant': example_assistant_7},
]
