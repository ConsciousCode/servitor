'''
GPT4All connector and completion.
'''

from gpt4all import GPT4All
from functools import cached_property

from . import Completion, Connector
from ..util import logger, async_await

logger.info("Import GPT4All connector")

class GPT4AllCompletion(Completion):
	'''A completion from GPT4All.'''
	
	def __init__(self, model, prompt, config):
		self.model = model
		self.prompt = prompt
		self.config = config
	
	async def __aiter__(self):
		yield self.sync()
	
	@async_await
	async def __await__(self):
		return self.sync()
	
	def __iter__(self):
		yield self.sync()
	
	def sync(self):
		# streaming refers to streaming to stdout
		return self.model.generator(self.prompt, streaming=False, **self.config)

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
	
	def __init__(self):
		super().__init__()
		self.models = {}
	
	@cached_property
	def model_list(self):
		return [normalize_name(name) for name in GPT4All.list_models()]
	
	def supports(self, config):
		return normalize_name(config['model']) in self.model_list
	
	def complete(self, prompt, config):
		model = config['model']
		
		if model not in self.models:
			name = model
			# All gpt4all model names are ggml-*.bin
			if not name.startswith("ggml-"):
				name = f"ggml-{name}"
			if not name.endswith(".bin"):
				name = f"{name}.bin"
			
			self.models[model] = GPT4All(name, config.get("model_path"), config.get("allow_download"))
				
		return GPT4AllCompletion(self.models[model], prompt, config)