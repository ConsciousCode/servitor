'''
Common definitions for language model endpoint connectors and completion
abstractions.
'''

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional, NamedTuple
import asyncio
import time
from .util import Registry, logger

class Delta(NamedTuple):
	text: str
	logprob: Optional[float]
	tokens: int

class Completion(ABC):
	'''
	A completion from a language model. Supports both streaming (via async
	iterators) and blocking (via awaitable).
	'''
	
	@abstractmethod
	def __aiter__(self) -> AsyncIterator[Delta]:
		'''Stream the completion one token at a time.'''
		pass
	
	@abstractmethod
	def __await__(self) -> Iterator[Delta]:
		'''Wait for the completion to finish.'''
		pass

class Connector(ABC):
	'''Generic connector for a language model endpoint.'''
	
	models: list[str]
	registry = Registry()
	
	def __init__(self, max_concurrent, rate, period):
		self.input_tokens = 0
		self.output_tokens = 0
		self.last_request = 0
		self.concurrent = asyncio.Semaphore(max_concurrent)
		self.times = [0]*rate
		self.period = period

	async def __aenter__(self):
		await self.concurrent.acquire()
		while True:
			t = time.monotonic()
			dt = t - (self.times[-1] + self.period)
			if dt > 0:
				break
			await asyncio.sleep(-dt)
		
		self.times.pop()
		self.times.insert(0, t)
	
	async def __aexit__(self, *args):
		self.concurrent.release()
	
	@staticmethod
	def register(name):
		def	decorator(cls):
			logger.debug(f"Registering connector {cls.__name__} as {name!r}")
			Connector.registry.register(name, cls.supports)
			return cls
		return decorator
	
	@classmethod
	def supports(cls, connector):
		if connector in cls.models:
			return cls()

Connector.register("openai")
class _openai_loader:
	'''Load openai connectors when they're actually requested.'''
	@staticmethod
	def supports(key):
		OPEN_AI_MODELS = [
			"text-curie-001", "text-babbage-001", "text-ada-001",
			"text-davinci-002", "text-davinci-003",
			"code-davinci-002",
			"gpt-3.5-turbo", "gpt-3.5-turbo-0301",
			"gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314",
			
			"davinci", "curie", "babbage", "ada"
		]
		if key in OPEN_AI_MODELS:
			from .openai import OpenAIConnector
			return OpenAIConnector