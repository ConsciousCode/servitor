'''
Public interface for LLM connectors, including the lazy loaders.
'''

import time
import asyncio
from contextlib import asynccontextmanager

from .connector import Completion, Connector, Throttle

# Lazy connector loaders - these will be replaced with the actual connector
#  when their modules are imported.

@Connector.register("openai")
class _openai_loader:
	'''Load openai connector when it's actually requested.'''
	@staticmethod
	def supports(key):
		from . import models
		if key in models.openai:
			from .openai import OpenAIConnector
			return OpenAIConnector

@Connector.register("gpt4all")
class _gpt4all_loader:
	'''Load gpt4all connector when it's actually requested.'''
	@staticmethod
	def supports(key):
		from . import models
		
		if key.startswith("ggml-"):
			key = key[5:]
		if key.endswith(".bin"):
			key = key[:-4]
		
		if key in models.gpt4all:
			from .gpt4all import GPT4AllConnector
			return GPT4AllConnector