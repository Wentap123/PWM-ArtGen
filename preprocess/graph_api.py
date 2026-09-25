"""Two-stage graph prediction; credentials and optional endpoint come from the environment."""
import base64
from io import BytesIO
import json
import os
import re

from . import prompts_graph, prompts_regions
from .parts import flatten_tree, validate_graph


def parse_response(text):
    if not text:
        raise ValueError('The graph API returned an empty response.')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, flags=re.I)
    try:
        value = json.loads(match.group(1) if match else text)
    except (TypeError, ValueError) as error:
        raise ValueError('The graph API response does not contain valid JSON.') from error
    if not isinstance(value, dict):
        raise ValueError('The graph API must return a JSON object.')
    return value


def request(image, prompt, model, graph=None):
    from openai import OpenAI
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('Set OPENAI_API_KEY, or provide cached graph/assignment JSON files.')
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    content = []
    if graph is not None:
        content.append({'type': 'text', 'text': 'Preserve this graph and child order exactly:\n' + json.dumps(graph)})
    content.append({'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}})
    messages = [{'role': 'system', 'content': prompt.system_prompt}]
    for example in prompt.examples:
        messages.extend([{'role': 'user', 'content': example['prompt']},
                         {'role': 'assistant', 'content': example['assistant']}])
    messages.append({'role': 'user', 'content': content})
    # No sampling overrides: reasoning models and compatible endpoints differ in support.
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                    base_url=os.environ.get('OPENAI_BASE_URL') or None,
                    timeout=180, max_retries=0)
    response = client.chat.completions.create(model=model, messages=messages)
    text = response.choices[0].message.content
    return parse_response(text), text


def predict_graph(image, model):
    tree, text = request(image, prompts_graph, model)
    graph = flatten_tree(tree)
    graph['original_response'] = text
    validate_graph(graph)
    return graph


def assign_regions(image, graph, model):
    value, text = request(image, prompts_regions, model, graph)
    return {'json': value, 'full_response': text}
