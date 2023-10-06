'''
Public interface for LLM connectors, including the lazy loaders.
'''

from .connector import Completion
from .connector import Throttle, Completion, Connector
from ..defaults import models

# Lazy connector loaders - these will be replaced with the actual connector
#  when their modules are imported.

@Connector.register("openai")
class _openai_loader(Connector):
	'''Load openai connector when it's actually requested.'''
	
	def supports(self, config):
		model = config.get("model")
		api_key = config.get("openai_api_key")
		
		if model and model in models.openai or api_key:
			from . import openai
			return True
		return False
	
	def complete(self, prompt, config):
		raise RuntimeError("Failed to load OpenAI connector.")

@Connector.register("gpt4all")
class _gpt4all_loader(Connector):
	'''Load gpt4all connector when it's actually requested.'''
	
	def supports(self, config):
		if model := config.get('model'):
			if model.startswith("ggml-"):
				model = model[5:]
			if model.endswith(".bin"):
				model = model[:-4]
			
			if model in models.gpt4all:
				from . import gpt4all
				return True
		return False
	
	def complete(self, prompt, config):
		raise RuntimeError("Failed to load GPT4All connector.")