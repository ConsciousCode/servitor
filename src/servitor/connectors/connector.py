'''
LLM provider connector base class.
'''

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, NamedTuple, TypeAlias, Protocol, Optional
import asyncio
from contextlib import contextmanager, asynccontextmanager
import time
from ..util import logger

class Throttle:
	'''
	Throttle LLM provider API requests by:
	* Concurrent requests
	* Requests per period
	* Tokens per period
	
	TODO: Support separate input/output token rates.
	'''
	
	def __init__(self, request_rate: float, token_rate: int, period: float=1, concurrent=3):
		'''
		Parameters:
			request_rate: Requests per period
			token_rate: Tokens per period
			period: Period in seconds
			concurrent: Concurrent requests
		'''
		self.request_rate = request_rate
		self.token_rate = token_rate
		self.token_allowance = token_rate
		self.request_allowance = request_rate
		self.period = period
		self.before = time.monotonic()
		self.concurrent = asyncio.Semaphore(concurrent)
	
	def can_proceed(self, input):
		now = time.monotonic()
		dt = now - self.before
		self.before = now
		self.token_allowance = min(self.token_rate, self.token_allowance + dt * self.token_rate / self.period)
		self.request_allowance = min(self.request_rate, self.request_allowance + dt * self.request_rate / self.period)
		if self.token_allowance >= input and self.request_allowance >= 1:
			self.token_allowance -= input
			self.request_allowance -= 1
			return True
		
		if self.token_allowance < input:
			logger.info(f"Throttling too many tokens {input} / {self.token_allowance}")
		if self.request_allowance < 1:
			logger.info("Throttling too many requests")
		
		return False
	
	@asynccontextmanager
	async def alock(self, input):
		'''Acquire a lock for the given number of input tokens (asynchronous).'''
		
		async with self.concurrent:
			while not self.can_proceed(input):
				await asyncio.sleep(0.1)
			yield

	@contextmanager
	def lock(self, input):
		'''Acquire a lock for the given number of input tokens (blocking).'''
		
		while not self.can_proceed(input):
			time.sleep(0.1)
		yield
	
	def output(self, tokens):
		'''Add output tokens to the allowance.'''
		self.token_allowance = min(self.token_rate, self.token_allowance + tokens)

class ModelConfig(NamedTuple):
	'''Configuration describing a model, its capabilities, and its modalities.'''
	
	name: str
	chat_only: bool
	context: int

class Completion(Protocol):
	'''
	A completion from a language model. Supports both streaming (via async
	iterators) and blocking (via awaitable).
	'''
	
	def __aiter__(self) -> AsyncIterator[str]:
		'''Stream the completion one token at a time.'''
	def __await__(self) -> Iterator[str]:
		'''Wait for the completion to finish.'''
	def __iter__(self) -> Iterator[str]:
		'''Blocking stream of the completion one token at a time.'''
	def sync(self) -> str:
		'''Use the blocking interface to return the completion as a string.'''

class Connector(ABC):
	'''Generic connector for a language model endpoint.'''
	
	registry: dict[str, 'Connector'] = {}
	
	@staticmethod
	def register(name):
		assert isinstance(name, str)
		
		def	decorator(cls):
			logger.debug(f"Registering connector {cls.__name__} as {name!r}")
			Connector.registry[name] = cls()
			return cls
		return decorator
	
	@classmethod
	def find(cls, config) -> 'Connector':
		for name, connector in cls.registry.items():
			if connector.supports(config):
				# Note: lazy loaders may update registry in supports()
				return cls.registry[name]
		raise KeyError(f"No connector found (model={config['model']!r})")
	
	@abstractmethod
	def supports(self, config) -> bool:
		'''Return a known connector which supports the given configuration, else None.'''
	
	@abstractmethod
	def complete(self, prompt, config) -> Completion:
		'''Build a completion from the given prompt and configuration.'''
