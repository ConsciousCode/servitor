# Servitor
Easily create "semantic functions" which use LLMs as a black box to execute the given task.

```python
>>> @semantic
... def list_people(text) -> list[str]:
...     """List people mentioned in the given text."""
...
>>> list_people("John and Mary went to the store.")
["John", "Mary"]
```

This project is licensed under the [MIT license](./LICENSE).

## Setup
```bash
$ pip install -r servitor/requirements.txt
$ pip install openai
```

```python
>>> from servitor import semantic
```

## Configuration
Auto-GPT style .env is supported by `DefaultKernel` and its singleton `semantic` with the following variables as defaults:
* `OPENAI_API_KEY` - OpenAI API key (currently required, later optional when other providers are added).
* `OPENAI_ORGANIZATION` - OpenAI organization ID for identification.
* `MODEL` - Model to use - searches registered Connectors for a model with this name. (default: `gpt-3.5-turbo`)
* `TEMPERATURE` - Temperature, "how random" it is from 0 to 2. (default: `0`)
* `TOP_P` - Top-P logit filtering. (default: `0.9`)
* `FREQUENCY_PENALTY` - Frequency penalty - penalizes repetition. (default: `0`)
* `PRESENCE_PENALTY` - Presence penalty - penalizes mentioning more than once. (default: `0`)
* `MAX_TOKENS` - Maximum number of tokens to return. (default: `1000`)
* `MAX_RETRY` - Maximum number of times to try fixing an unparseable completion. (default: `3`)
* `MAX_RATE` - Maximum number of requests per period. (default: `60`)
* `MAX_PERIOD` - Period in seconds. (default: `60`)

`Kernel` requires a `Connector` and an `Adapter` to be passed to its constructor.

In addition, `Kernel` instantiation and decorating can take either a `config` kw-only argument or kwargs with these names:
* `api_key`
* `model`
* `temperature`
* `top_p`
* `frequency_penalty`
* `presence_penalty`
* `max_tokens`
* `max_retry`
* `max_rate`
* `max_period`

## How it works
Servitor has the following components:
* **Adapter** - Interface between the natural language of the LLM and the programmatic logic of normal Python.
* **Connector** - Abstract interface for LLM provider. Currently only OpenAI is supported, but it's been designed for adding new connectors without an `openai` library dependency.
* **Completion** - A chat/text completion from an LLM.
* **Kernel** - A set of defaults and glue logic which make semantic functions work.
* **Task** - A textual task for the LLM to complete.
* **Semantic function** - A function which uses an LLM to execute a task.

First, you instantiate a kernel (`semantic` is provided for ease-of-use). Then, the kernel instance can either be called on a string (used as the prompt) or as a decorator on a function. It will call the function; if the result is None, it uses the docstring. Otherwise, it uses the result as the task. The adapter converts the task to the full prompt for the LLM and implements error-recovery logic to ensure the result is parsed into a machine-readable format. With all this, you can define a semantic function with a simple decorator, and it looks and acts exactly like a normal (async) function.

More advanced adapters use the type annotations of the function to both prompt and validate the responses of the LLM. The ones provided use HJSON (Human-JSON) which is a generalization of JSON to be more permissive of the LLM's generation, which tend to contain small parsing errors, and it typically uses many fewer tokens.

### More details
The adapter is a bidirectional generator which yields prompts, receives responses, and returns a final answer. The implementation of `Adapter` is expected to take care of the vast majority of cases, but this interface is very general to allow for more complex use-cases. The connector is simply an async callable which returns a `Completion` object. This is a thin wrapper around the raw response from the LLM, which can be used as an async iterator for streaming or as an awaitable for blocking. Currently only blocking is actually used.

## Available classes
### Adapters
All adapters are contained within `adapter.py`. Anywhere an adapter is expected, you can pass a registered string instead.
* `Adapter` - (not registered) Abstract base class for adapters.
* `TaskAdapter` - (`"task"`) Simplest adapter, just passes the prompt through to the LLM and returns the result.
* `PlainAdapter` - (`"plain"`) Prompts the LLM to give an answer rather than simply complete, but has no parsing. Mostly used as a base class for more advanced adapters.
* `TypeAdapter` - (`"type"`) Uses type annotations and HJSON to prompt and parse the result.
* `ChainOfThoughtAdapter` - (`"cot"`) Uses Chain of Thought prompting to get a more coherent response. It also wraps the result in a `ChainOfThought` named tuple `(thoughts, answer)`.

### Connectors
The common classes for connectors are in `complete.py`, but each connector has its own file which is loaded dynamically to avoid unused dependencies.
* `Connector` - Abstract base class for connectors.
* `Completion` - Wrapper for the raw response of an LLM provider.
* `Delta` - Used by completions to indicate new generated tokens.
* `openai.OpenAIConnector` - Connector for OpenAI's API.
* `openai.OpenAICompletion` - Wrapper for OpenAI's completion response.

### Kernels
All kernels are in `kernel.py`. Only two are defined:
* `Kernel` - Simple base class for kernels.
* `DefaultKernel` - Kernel implementation loading defaults from config files and ENV. It has an instance `semantic` for ease-of-use.

Kernels can be called, used as decorators (with optional arguments), or they can do an ordinary completion using `Adapter.complete`

## Examples
You can use the `semantic` singleton to define a semantic function:
```python
>>> @semantic
... def list_people(text) -> list[str]:
...     """List people mentioned in the given text."""
...
>>> list_people("John and Mary went to the store.")
["John", "Mary"]
```

You can customize the prompt by returning a string from the function:
```python
>>> @semantic
... def classify_valence(text: str) -> float:
...     return """Classify the valence of the given text as a value between -1 and 1."""
...
>>> classify_valence("I am happy.")
0.9
```

The kernel decorator can be used on classes, wrapping the `__call__` method.
```python
>>> @semantic
... class SelectTool:
...     def __init__(self, tools: list[Callable]):
...         self.tools = tools
...
...     def __call__(self, task: str) -> str:
...         """Select a tool from the given list."""
...         return f"Select a tool for the given task from this list: {list(self.tools.keys())}"
```

Plain adapter is good for simple text to text tasks:
```python
>>> @semantic(adapter="plain")
... def summarize(text) -> str:
...     """Summarize the given text in two sentences or less."""
```

Dumber models can be used for dumber tasks:
```python
>>> @semantic(model="text-ada-002", adapter="none")
... def what_is_this(text):
...     return f"{text}\n\nWhat is this?"
...
>>> what_is_this("lorem ipsum...")
"This is an example of placeholder Latin text, commonly known as Lorem Ipsum."
```

## Synchrony
Servitor uses async internally, but the semantic function is properly wrapped to have the same type of synchrony as its definition. To retain async, make its definition async:

```python
>>> @semantic(adapter="cot")
... async def summarize(concept: str) -> str:
... 	"""Summarize the given text in two sentences or less."""
...
>>> asyncio.run(summarize("long text..."))
"A summary."
```

## What's the point?
I've noticed a trend of people using LLMs as semi-personified agents. They're given personalities, presented with tools to select from and plan with (eg Microsoft Semantic Kernels), and generally expected to act like some kind of proto-person. This is understandable given the first truly usable instance we saw was ChatGPT, a chatbot with a very simple cognitive architecture (rolling, discrete, turn-based chat logs with an initial system prompt), however it severely limits the usability and composability that's possible. This library is intended to reframe LLMs as what they are - very good stochastic models of language - and thus treat them as a kind of natural language inference engine, which is what they're really designed for. By reifying tasks as discrete functions which can be composed with ordinary logic, programs can be given a sort of embedded intelligence which is otherwise impossible.

## What is a "servitor"?
A "[servitor](https://en.wikipedia.org/wiki/Servitor_(chaos_magic))" is a kind of artificial spirit in [chaos magick](https://en.wikipedia.org/wiki/Chaos_magic). Essentially, you create a miniature proto-consciousness which is given a specific task to perform autonomously. This might be something as simple as reminding you to wake up at a certain time, or as complex as performing a ritual on the "astral plane" on your behalf. Whether or not you believe in all that, the parallels are clear: It's a kind of AI program within your own mind which completes exactly one very specific task given to it in natural language. LLMs are more general than servitors, but restricting their capacities allows them to excel at what they're best at rather than trying to meet a standard they're not yet capable of. It's my opinion that AGI will not be a single monolithic LLM, but rather a cognitive architecture system which uses LLMs as computational units. Think of the cognitive architecture as the [Chinese Room](https://en.wikipedia.org/wiki/Chinese_room) and the LLMs as the person inside. Neither understands Chinese, but the total system does.