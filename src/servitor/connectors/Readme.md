[Main](../../) / [Back](../)

Connectors for various LLM providers. Putting these in separate files allows us to load them dynamically, avoiding unnecessary load times and dependencies. __init__.py provides some dummy loaders so you can easily import models from any of the connectors without explicitly loading the library, but these are static lists of supported models and so are generally going to be out of date.

Safe and "correct" usage:
```python
from servitor import semantic
import servitor.connectors.openai

@semantic(model="text-davinci-003")
def func():
	"""Etcetera"""
```

Easy and convenient usage:
```python
from servitor import semantic

@semantic(model="text-davinci-003")
def func():
	"""Etcetera"""
```

In the second example, we must assume the provided model is in the static model directory so it knows to load the openai connector. This is not a problem for the first example, as the connector is loaded explicitly.