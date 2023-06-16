'''
LLM provider connector base class.
'''

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator
import asyncio
from contextlib import asynccontextmanager
import time
from ..util import Registry, logger

class Throttle:
	'''
	Throttle LLM provider API requests by:
	* Concurrent requests
	* Requests per period
	* Tokens per period
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
	
	@asynccontextmanager
	async def __call__(self, tokens):
		if self.concurrent.locked():
			logger.info("Throttling concurrent requests")
		
		async with self.concurrent:
			while True:
				now = time.monotonic()
				dt = now - self.before
				self.before = now
				self.token_allowance = min(self.token_rate, self.token_allowance + dt * self.token_rate / self.period)
				self.request_allowance = min(self.request_rate, self.request_allowance + dt * self.request_rate / self.period)
				if self.token_allowance >= tokens and self.request_allowance >= 1:
					self.token_allowance -= tokens
					self.request_allowance -= 1
					break
				
				if self.token_allowance < tokens:
					logger.info(f"Throttling too many tokens {tokens} / {self.token_allowance}")
				if self.request_allowance < 1:
					logger.info("Throttling too many requests")
				await asyncio.sleep(0.1)
			
			yield
	
	def add_tokens(self, tokens):
		'''Add tokens to the allowance.'''
		self.token_allowance = min(self.token_rate, self.token_allowance + tokens)

class Completion(ABC):
	'''
	A completion from a language model. Supports both streaming (via async
	iterators) and blocking (via awaitable).
	'''
	
	@abstractmethod
	def __aiter__(self) -> AsyncIterator[str]:
		'''Stream the completion one token at a time.'''
		pass
	
	@abstractmethod
	def __await__(self) -> Iterator[str]:
		'''Wait for the completion to finish.'''
		pass

class Connector(ABC):
	'''Generic connector for a language model endpoint.'''
	
	models: list[str]
	registry = Registry()
	
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