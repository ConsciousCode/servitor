'''
Common utilities.
'''

from typing import TypeVar, Optional, Callable
import inspect
import logging
import os

logger = logging.getLogger("servitor")
logger.setLevel(os.getenv("LOG_LEVEL", "WARNING").upper())

T = TypeVar("T")
def default(x: Optional[T], y: T|Callable[[], T]) -> T:
	'''Extensible defaults for function arguments.'''
	return x if x is not None else y() if callable(y) else y

class ParseError(RuntimeError):
	'''LLM parsing error.'''
	
	def __init__(self, msg, attempts: list[tuple[str, Exception]]):
		'''
		Parameters:
			msg: Error message
			attempts: List of attempts (response-error pairs)
		'''
		super().__init__(msg)
		self.attempts = attempts

class ThrottleError(ConnectionError):
	'''Error raised when the LLM endpoint is throttling.'''
	pass

class BusyError(ConnectionError):
	'''Error raised when the LLM endpoint is busy.'''
	pass

class Registry:
	'''Generic registry for mapping names to objects.'''
	
	def __init__(self):
		self.registry = {}
	
	def find(self, key):
		if isinstance(key, str):
			for k, supports in self.registry.items():
				if x := supports(key):
					return x
			raise KeyError(key)
		else:
			return key
	
	def register(self, name, supports):
		self.registry[name] = supports

def build_task(origin, args, kwargs):
	'''
	Common function for building a task from a string or function. Uses the
	origin itself if it's a string, the return, or the docstring.
	'''
	
	if callable(origin):
		if task := origin(*args, **kwargs):
			return task
		return inspect.getdoc(origin)
	return origin