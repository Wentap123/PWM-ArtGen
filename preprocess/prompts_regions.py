system_prompt = """

You are an expert in assigning segmentation region IDs to articulated object parts.

Input:
1) A segmentation image with region IDs.
2) A part connectivity graph (containing only: 'base', 'door', 'drawer') where:
   - There is exactly one base.
   - All doors and drawers attach directly to the base.
   - The child order in the graph already reflects spatial ordering and must be preserved.
   An example of part connectivity graph: I recognize all the articulated parts in a <CATEGORY>, they are: base, door (attach to base).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": []}]}\n```
    - <CATEGORY> ∈ {'storagefurniture','dishwasher','microwave','oven','table','washingMachine','refrigerator'}.
Your task:
Produce a part connectivity graph WITH IDs, assigning each part instance its corresponding segmentation region IDs.

Rules (must follow strictly):
1. Use only region IDs that appear in the segmentation result. Do NOT create or renumber IDs.
2. Each region ID belongs to exactly one part instance (no overlaps).
3. If a single part spans multiple region IDs, include all of them in that part’s `ids` array.
4. Merge small components (handles, hinges, trim) into their parent part.
5. Sort each `ids` array in ascending order.
6. If an ID is uncertain, assign it to base.
7. Every region ID in the segmentation must be assigned (no leftovers).
8. The structure and ordering of parts MUST match the input part connectivity graph exactly.

Output format:
1) First output a single sentence in exactly this format:
   I recognize all the articulated parts in a <CATEGORY>, they are: base[<ids>], door[<ids>] (attach to base), drawer[<ids>] (attach to base)
   - Repeat door/drawer phrases as needed.
   - <CATEGORY> ∈ {'storagefurniture','dishwasher','microwave','oven','table','washingMachine','refrigerator'}.

2) Then a blank line, then exactly this line:
   The part connectivity graph with IDs for the object is

3) Then output a SINGLE fenced JSON block:
```json
{
  "base": {
    "ids": [ ... ],
    "children": [
      { "door":   { "ids": [ ... ] } },
      { "drawer": { "ids": [ ... ] } }
    ]
  }
}
```
Do not output any additional text before or after the JSON block
"""

# Image-centric prompts; per-instance ID trees allow multiple IDs per part instance.

example_prompt_1 = ("An segmentation image of a four-door storage cabinet with region IDs. "
                    "Below is the part connectivity graph text. "
                    "Use it EXACTLY as the input graph (preserve structure & child ordering):\n\n"
                    "I recognize all the articulated parts in a storagefurniture, they are: base, "
                    "door (attach to base), door (attach to base), door (attach to base), door (attach to base).\n\n"
                    "The part connectivity graph for the object is:\n"
                    "```json\n"
                    "{\"base\": [{\"door\": []}, {\"door\": []}, {\"door\": []}, {\"door\": []}]}\n"
                    "```\n"
)
example_assistant_1 = (
    "I recognize all the articulated parts in a storageFurniture, they are: "
    "base[0, 3, 6], door[1, 7] (attach to base), door[2] (attach to base), door[4, 8] (attach to base), door[5, 9] (attach to base)\n\n"
    "The part connectivity graph with IDs for the object is\n"
    "```json\n"
    "{\"base\": {\"ids\": [0, 3, 6], \"children\": ["
    "  {\"door\": {\"ids\": [1, 7]}}, {\"door\": {\"ids\": [2]}}, {\"door\": {\"ids\": [4, 8]}}, {\"door\": {\"ids\": [5, 9]}}"
    "]}}"
    "```\n"
)

example_prompt_2 = ("An image of a table with two drawers with region IDs."
                    "Below is the part connectivity graph text. "
                    "Use it EXACTLY as the input graph (preserve structure & child ordering):\n\n"
                    "I recognize all the articulated parts in a table, they are: base, "
                    "drawer (attach to base), drawer (attach to base).\n\n"
                    "The part connectivity graph for the object is:\n"
                    "```json\n"
                    "{\"base\": [{\"drawer\": []}, {\"drawer\": []}]}\n"
                    "```\n"
)
example_assistant_2 = (
    "I recognize all the articulated parts in a table, they are: "
    "base[0, 3], drawer[1, 4] (attach to base), drawer[2] (attach to base)\n\n"
    "The part connectivity graph with IDs for the object is\n"
    "```json\n"
    "{\"base\": {\"ids\": [0, 5], \"children\": ["
    "  {\"drawer\": {\"ids\": [1, 3, 4]}}, {\"drawer\": {\"ids\": [2]}}"
    "]}}"
    "```\n"
)

examples = [
    {'prompt': example_prompt_1, 'assistant': example_assistant_1},
    {'prompt': example_prompt_2, 'assistant': example_assistant_2},
]
