'''
LLM provider connector base class.
'''

import asyncio
from contextlib import contextmanager, asynccontextmanager
import time
from ..util import logger
from ..typings import ABC, abstractmethod, AsyncIterator, Iterator, NamedTuple, TypeAlias, Protocol, Optional

class Throttle:
	'''
	Throttle LLM provider API requests by:
	* Concurrent requests
	* Requests per period
	* Tokens per period
	
	TODO: Support separate input/output token rates.
	'''

	request_rate: float
	'''Maximum requests per period allowed.'''
	token_rate: int
	'''Maximum tokens per period allowed.'''
	token_allowance: int
	'''How many tokens have been seen in the last period.'''
	request_allowance: float
	'''How many requests have been made in the last period.'''
	period: float
	'''Period in seconds.'''
	wait: float
	'''How often to sleep for when throttling.'''
	before: float
	'''Previous monotonic timestamp.'''
	concurrent: asyncio.Semaphore
	'''Concurrent request semaphore.'''
	
	def __init__(self, request_rate: float, token_rate: int, period: float=1, concurrent=3, wait=0.1):
		'''
		Parameters:
			request_rate: Requests per period
			token_rate: Tokens per period
			period: Period in seconds
			concurrent: Concurrent requests
			wait: 
		'''
		self.request_rate = request_rate
		self.token_rate = token_rate
		self.token_allowance = token_rate
		self.request_allowance = request_rate
		self.period = period
		self.wait = wait
		self.before = time.monotonic()
		self.concurrent = asyncio.Semaphore(concurrent)
	
	def update_allowance(self):
		'''Update allowances.'''
		now = time.monotonic()
		dt = now - self.before
		self.before = now
		self.token_allowance = min(self.token_rate, self.token_allowance + dt * self.token_rate / self.period)
		self.request_allowance = min(self.request_rate, self.request_allowance + dt * self.request_rate / self.period)
		print(f"update_allowance(): {self.token_allowance=} {self.request_allowance=}")
	
	def can_proceed(self, tokens: int) -> bool:
		'''Whether or not it's ok to send a new request with the given token count.'''
		print(f"can_proceed({tokens})")
		self.update_allowance()
		if self.token_allowance >= tokens and self.request_allowance >= 1:
			self.token_allowance -= tokens
			self.request_allowance -= 1
			return True
		
		if self.token_allowance < tokens:
			logger.info(f"Throttling too many tokens {tokens} / {self.token_allowance}")
		if self.request_allowance < 1:
			logger.info("Throttling too many requests")
		
		return False
	
	@asynccontextmanager
	async def alock(self, tokens: int):
		'''Acquire a lock for the given number of input tokens (asynchronous).'''
		
		async with self.concurrent:
			while not self.can_proceed(tokens):
				await asyncio.sleep(self.wait)
			yield

	@contextmanager
	def lock(self, tokens: int):
		'''Acquire a lock for the given number of input tokens (blocking).'''
		
		while not self.can_proceed(tokens):
			time.sleep(self.wait)
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
	def __call__(self) -> str:
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
