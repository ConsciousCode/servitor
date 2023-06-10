'''
@module servitor

Semantic functions for LLMs.

Allows the trivial creation of "semantic functions" using decorators, which 
use LLMs as a black box to execute the given task.

Examples:

>>> semantic.__call__ is semantic.function
True
>>> @semantic
... def list_people(text) -> list[str]:
...	 """List people mentioned in the given text."""
...
>>> await list_people("John and Mary went to the store.")
["John", "Mary"]

If non-None is returned, it is used as the prompt for the LLM instead of
the function's docstring:
>>> @semantic
... def classify_valence(text: str) -> float:
...	return """Classify the valence of the given text as a value between -1 and 1."""
...
>>> await classify_valence("I am happy.")
0.9

Tasks are reified completions which don't include any processing such as type
annotations, argument passing, or return value processing. They are useful for
simple natural language processing tasks or for heavy customization. They return
Completion objects, which are async-iterators for streaming and awaitables for
blocking:
>>> @semantic.task
... def what_is_this(text):
...	 return f"{text}\n\nWhat is this?"
...
>>> summarize("lorem ipsum...")
Completion()
>>> await _
"This is an example of placeholder Latin text, commonly known as Lorem Ipsum."

Semantic funtions can take arguments passed to the LLM, which can be used to customize
the model used, temperature, best-of, top-p, frequency penalty, etc:
>>> @semantic(temperature=0)
... def spellcheck(text) -> str:
...	 """Spellcheck the given text, returning correctly spelled text."""
...
>>> await spellcheck("I am happpy.")
"I am happy."

"Connectors" are abstract interfaces to LLM completion services.

"Adapters" are coroutines which yield prompts, receive responses, and return the
parsed result. They are the interface used to translate between computational functions
and natural language used by LLMs. They are responsible for providing context in the
form of prompts, parsing the response into a machine-readable format, and handling
the correction of errors made by the LLM (which frequently fail to output perfectly
formatted text, so adapters prompt them to fix these mistakes).

Adapters handle all error handling, including timeouts, throttling, and parsing errors.

"Kernels" are the set of defaults and glue logic which make semantic functions work.
Once instantiated, they can be used to create new semantic functions.

`DefaultKernel` and its singleton `semantic` are provided for convenience and cover
most use-cases. They check for common environment variables and config files to
automatically determine the best adapter and connector to use. They're also zero-cost,
so they won't touch the filesystem unless they're used.
'''

from .kernel import Kernel, DefaultKernel
from .adapter import Adapter, TaskAdapter, PlainAdapter, TypeAdapter, ChainOfThoughtAdapter, ChainOfThought
from .complete import Connector, Completion, Delta
from .util import ParseError, ThrottleError, BusyError

semantic = DefaultKernel()