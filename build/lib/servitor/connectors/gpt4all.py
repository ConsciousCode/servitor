'''
GPT4All connector and completion.
'''

from gpt4all import GPT4All
from typing import Optional

from . import Completion, Connector
from ..util import logger

logger.info("import", extra={"module": "gpt4all"})

class GPT4AllCompletion(Completion):
	'''A completion from GPT4All.'''
	
	def __init__(self, connector, prompt, config):
		self.connector = connector
		self.prompt = prompt
		self.config = config
	
	async def __aiter__(self):
		async with self.connector:
			async for item in self.connector.model.generator(self.prompt, streaming=True, **self.config):
				yield item
	
	def __await__(self):
		yield from self.connector.__aenter__().__await__()
		try:
			return self.connector.model.generate(self.prompt, streaming=False, **self.config)
		finally:
			yield from self.connector.__aexit__(None, None, None).__await__()

def normalize_name(name):
	'''Normalize GPT4All model names.'''
	
	name = name.lower()
	if name.startswith("ggml-"):
		name = name[5:]
	if name.endswith(".bin"):
		name = name[:-4]
	return name

@Connector.register("gpt4all")
class GPT4AllConnector(Connector):
	'''Connector for OpenAI API.'''
	
	models = None
	
	def __init__(self, *, model: Optional[str]=None, model_path: Optional[str]=None, allow_download=True, **other):
		'''
		Parameters:
			model: Model to use.
		'''
		super().__init__()
		
		if model is not None:
			if not model.startswith("ggml-"):
				model = f"ggml-{model}"
			if not model.endswith(".bin"):
				model = f"{model}.bin"
		
		self.model = GPT4All(model_name=model, model_path=model_path, allow_download=allow_download)
	
	def __call__(self, prompt, **kwargs):
		return GPT4AllCompletion(self, prompt, {
			"model": self.model,
			"temperature": kwargs.get("temperature"),
			"max_tokens": kwargs.get("max_tokens"),
			"top_p": kwargs.get("top_p"),
			"frequency_penalty": kwargs.get("frequency_penalty"),
			"presence_penalty": kwargs.get("presence_penalty"),
			"stop": kwargs.get("stop"),
			"stream": kwargs.get("stream"),
			# openai library doesn't recognize this?
			#"best_of": kwargs.get("best_of")
		})
	
	@classmethod
	def supports(cls, key):
		if cls.models is None:
			cls.models = [normalize_name(name) for name in GPT4All.list_models()]
		
		return normalize_name(key) in cls.models