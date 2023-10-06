'''
Kernels aggregate the machinery and defaults needed to create, coordinate, and
execute semantic functions. Unlike MS Semantic Kernels, they do not store any
semantic functions themselves to be exposed to a single LLM instance typingto use as
tools, though this sort of functionality can be built *on top* of kernels.
'''

from functools import wraps, cached_property
from typing import Optional
from collections.abc import Callable
from types import NoneType
import inspect

from .typings import KernelConfig, SemanticOrigin
from .config import DefaultConfig
from .util import default, logger, parse_bool
from .adapter import Adapter, TypeAdapter, AdapterRef
from .connectors import Completion, Connector
from . import defaults

def force_sync(fn):
	'''Make a non-awaiting async function synchronous.'''

	def invoke_dummy_async(*args, **kwargs):
		'''Invoke the dummy async function.'''
		coro = fn(*args, **kwargs).__await__()
		try:
			# send() should immediately raise StopIteration
			if coro.send(None) is not None:
				raise RuntimeError("Semantic function definitions cannot await!")
			raise RuntimeError("Malformed awaitable did not stop iteration.")
		except StopIteration as e:
			return e.value
	
	if inspect.iscoroutinefunction(fn):
		return wraps(fn)(invoke_dummy_async)
	return fn

class SemanticFunction:
	'''Natural language semantic function class.'''

	_connector: Connector
	'''Connects to an LLM.'''
	_adapter: Adapter
	'''Adapts code to/from LLM plaintext.'''
	_origin: SemanticOrigin
	'''Origin potentially wrapped in a function which returns the task.'''
	_async: bool
	'''Whether or not the semantic function is asynchronous.'''

	config: KernelConfig
	'''Local copy of kernel configuration.'''

	def __init__(self, origin, connector, adapter, config):
		self._connector = connector
		self._adapter = adapter
		self._async = inspect.iscoroutinefunction(origin)
		self._origin = force_sync(origin)
		self.config = config

		# Wraps is mutating because it expects to be an annotation
		wraps(origin)(self)
	
	def __call__(self, *args, **kwargs):
		if self._async:
			@wraps(self._origin)
			async def invoke_async_semfn(*args, **kwargs):
				'''Semantic function closure.'''
				try:
					task = self._adapter(self._origin, *args, **kwargs)
					prompt = next(task)
					while prompt:
						completion = self._connector.complete(prompt, self.config)
						prompt = task.send(await completion)
				except StopIteration as e:
					return e.value
			
			# return the instantiated coroutine
			return invoke_async_semfn(*args, **kwargs)
		
		# Synchronous invocation
		try:
			task = self._adapter(self._origin, *args, **kwargs)
			prompt = next(task)
			while prompt:
				completion = self._connector.complete(prompt, self.config)
				prompt = task.send(completion())
		except StopIteration as e:
			return e.value
	
	def __str__(self):
		return f"<semantic {self.__name__} at {id(self)}>"

class Kernel:
	'''
	Aggregates machinery needed to execute and coordinate semantic functions.
	'''
	
	def __init__(self, adapter: AdapterRef, config: Optional[KernelConfig]=None, **kwargs):
		'''
		Parameters:
			adapter: Parsing and session maintenance for the LLM's responses.
			config: Default configuration for connector (merged with kwargs).
			name: Optional name for the kernel, used for error messages.
		'''
		self.adapter = Adapter.registry.find(adapter)
		self.config = {**defaults.config, **(config or {}), **kwargs}
	
	def complete(self, prompt: str, config: Optional[KernelConfig]=None, **kwargs) -> Completion:
		'''Normal completion.'''
		
		config = {**self.config, **(config or {}), **kwargs}
		connector = Connector.find(config)
		return connector.complete(prompt, config)
		
	def __call__(self,
	    	origin: Optional[SemanticOrigin]=None,
		    adapter: Optional[AdapterRef]=None,
		    *,
		    config: Optional[KernelConfig]=None,
		    **kwargs
		):
		'''
		Build a semantic function from its metadata. Also works as a decorator.
		
		Parameters:
			origin: Origin of the semantic function, a string, a class, or a function.
			adapter: Adapter to use for this function.
			config: Configuration for this function.
			kwargs: Additional configuration for this function (merged with config).
		'''
		
		if not isinstance(origin, (str, type, Callable, NoneType)):
			raise TypeError(f"Origin {origin!r} must be a string, class, or callable.")
		
		if not isinstance(adapter, (str, Adapter, Callable, NoneType)):
			raise TypeError(f"Adapter {adapter!r} must be a string, adapter, or coroutine.")
		
		# Combine defaults, config keyword, and uncaught keywords into one config
		config = DefaultConfig({**(config or {}), **kwargs}, self.config)
		
		if isinstance(adapter, str):
			adapter = Adapter.find(adapter)(config)
		else:
			adapter = self.adapter
		connector = Connector.find(config)
		
		def decorator(origin):
			'''Semantic function decorator.'''
			
			# Decorating class wraps the __call__ method
			if inspect.isclass(origin):
				if hasattr(origin, "__call__"):
					origin.__call__ = decorator(origin.__call__)
					return origin
				raise TypeError(f"Task {origin.__name__} instances must be callable.")
			
			return SemanticFunction(origin, connector, adapter, config)
		
		# Apply decorator if we know the origin
		return decorator(origin) if origin else decorator

class DefaultKernel(Kernel):
	'''Kernel with reasonable defaults for quick use. Does nothing until used.'''
	
	def __init__(self):
		# no super().__init__() to avoid overwriting properties
		pass
	
	@cached_property
	def config(self):
		logger.info("DefaultKernel used")
		
		import os
		from dotenv import load_dotenv
		load_dotenv()

		from .config import CONFIG_SCHEMA
		
		config = {}
		for name, schema in CONFIG_SCHEMA.items():
			value = os.getenv(name.upper()) or defaults.config.get(name)
			if value is not None and value != "":
				config[name] = schema(value)
		
		logger.debug(f"Loaded {config=}")
		return config
	
	@cached_property
	def adapter(self):
		return TypeAdapter(self.config)