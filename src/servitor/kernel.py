'''
Kernels aggregate the machinery and defaults needed to create, coordinate, and
execute semantic functions. Unlike MS Semantic Kernels, they do not store any
semantic functions themselves to be exposed to a single LLM instance to use as
tools, though this sort of functionality can be built *on top* of kernels.
'''

from functools import wraps, cached_property
import typing
from typing import Optional, Callable, TypeAlias, Protocol, TypedDict, TypeVar, ParamSpec, Awaitable
from types import NoneType
import inspect

from .util import default, logger, parse_bool
from .adapter import Adapter, TypeAdapter, AdapterRef
from .connectors import Completion, Connector
from . import defaults

NotRequired = getattr(typing, "NotRequired", Optional)

P = ParamSpec("P")
R = TypeVar("R")

class SyncDocstringSemanticFunction(Protocol):
	'''Semantic function built using its docstring.'''
	__doc__: str
	def __call__(self, *args: P.args, **kwargs: P.kwargs) -> NoneType: ...
SyncSemanticOrigin: TypeAlias = str|SyncDocstringSemanticFunction|Callable[P, str]

class AsyncDocstringSemanticFunction(Protocol):
	'''Semantic function built using its docstring.'''
	__doc__: str
	def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[None]: ...
AsyncSemanticOrigin: TypeAlias = AsyncDocstringSemanticFunction|Callable[P, Awaitable[str]]

SyncSemanticFunction: TypeAlias = Callable[P, R]
AsyncSemanticFunction: TypeAlias = Callable[P, Awaitable[R]]

class KernelConfig(TypedDict, total=False):
	'''Configuration for a kernel.'''
	
	open_api_key: NotRequired[str]
	'''API key for some providers.'''
	model: NotRequired[str]
	'''Model name for the connector.'''
	temperature: NotRequired[float]
	'''Temperature for gerating completions.'''
	top_p: NotRequired[float]
	'''Top-P logit filtering.'''
	top_k: NotRequired[int]
	'''Top-K logit filtering - not always supported.'''
	frequency_penalty: NotRequired[float]
	'''Frequency penalty for generating completions.'''
	presence_penalty: NotRequired[float]
	'''Presence penalty for generating completions.'''
	max_tokens: NotRequired[int]
	'''Maximum number of tokens to generate.'''
	retry: NotRequired[int]
	'''Number of times to retry a request.'''
	concurrent: NotRequired[int]
	'''Maximum number of concurrent requests to allow (if connector supports throttling).'''
	request_rate: NotRequired[float]
	'''Number of requests per period (if connector supports throttling).'''
	token_rate: NotRequired[float]
	'''Number of tokens per period (if connector supports throttling).'''
	period: NotRequired[float]
	'''Period for request and token rate (if connector supports throttling).'''

def _build_async_semantic_fn(origin, connector, adapter, config) -> AsyncSemanticFunction:
	'''Build an async semantic function from preconstructed parameters.'''
	
	# Origin is async to indicate synchrony, but doesn't await anything.
	#  Adapter expects a synchronous function, so we have to wrap it.
	@wraps(origin)
	def sync_origin(*args, **kwargs):
		'''Make a non-awaiting async function synchronous.'''
		
		coro = origin(*args, **kwargs).__await__()
		try:
			if coro.send(None) is not None:
				raise RuntimeError("Semantic function definitions cannot await!")
			raise RuntimeError("Malformed awaitable did not stop iteration.")
		except StopIteration as e:
			return e.value
	
	@wraps(origin)
	async def async_semantic_fn(*args, **kwargs):
		'''Semantic function closure.'''
		
		try:
			task = adapter(sync_origin, *args, **kwargs)
			prompt = next(task)
			while prompt:
				completion = connector.complete(prompt, config)
				prompt = task.send(await completion)
		except StopIteration as e:
			return e.value
	
	return async_semantic_fn

def _build_sync_semantic_fn(origin, connector, adapter, config) -> SyncSemanticFunction:
	'''Build a synchronous semantic function from preconstructed parameters.'''
	
	def sync_semantic_fn(*args, **kwargs):
		'''Semantic function closure.'''
		
		try:
			task = adapter(origin, *args, **kwargs)
			prompt = next(task)
			while prompt:
				completion = connector.complete(prompt, config)
				prompt = task.send(completion.sync())
		except StopIteration as e:
			return e.value
	
	# Could be string or function
	if callable(origin):
		return wraps(origin)(sync_semantic_fn)
	return sync_semantic_fn

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
	    	origin: Optional[SyncSemanticOrigin|AsyncSemanticOrigin]=None,
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
		
		config = {**self.config, **(config or {}), **kwargs}
		
		if not isinstance(origin, (str, type, Callable, NoneType)):
			raise TypeError(f"Origin {origin!r} must be a string, class, or function.")
		
		if not isinstance(adapter, (str, Adapter, Callable, NoneType)):
			raise TypeError(f"Adapter {adapter!r} must be a string, adapter, or coroutine.")
		
		adapter = Adapter.find(adapter or self.adapter)
		connector = Connector.find(config)
		
		def decorator(origin):
			'''Semantic function decorator.'''
			
			# Decorating class wraps the __call__ method
			if inspect.isclass(origin):
				if hasattr(origin, "__call__"):
					call = decorator(origin.__call__)
					origin.__call__ = call
					call.__name__ = f"{origin.__name__}.__call__"
					
					return origin
				raise TypeError(f"Task {origin} instances must be callable.")
			
			# Build the semantic function with the right synchrony
			if inspect.iscoroutinefunction(origin):
				return _build_async_semantic_fn(origin, connector, adapter, config)
			else:
				return _build_sync_semantic_fn(origin, connector, adapter, config)
		
		# Apply decorator if we know the origin
		return decorator(origin) if origin else decorator

def clamp(t, lo, hi):
	'''Closure to clamp between lo and hi.'''
	return lambda v: max(lo, min(hi, t(v)))
inf = float("inf")

# Configuration schema
CONFIG_SCHEMA = dict(
	# OpenAI
	openai_api_key = str,
	openai_organization = str,
	
	# GPT4All
	model_path = str,
	allow_download = parse_bool,
	
	# Common parameters
	model = str,
	
	# Generation parameters
	temperature = clamp(float, 0, 2),
	top_p = clamp(float, 0, 1),
	top_k = clamp(int, 0, inf),
	frequency_penalty = clamp(float, 0, 1),
	presence_penalty = clamp(float, 0, 1),
	max_tokens = clamp(int, 0, inf),
	best_of = clamp(int, 1, inf),
	
	# Throttling
	retry = clamp(int, 0, inf),
	concurrent = clamp(int, 1, inf),
	request_rate = clamp(int, 0, inf),
	token_rate = clamp(int, 0, inf),
	period = clamp(float, 0, inf)
)

class DefaultKernel(Kernel):
	'''Kernel with reasonable defaults for quick use. Does nothing until used.'''
	
	__name__ = "semantic"
	
	def __init__(self):
		# no super().__init__() to avoid overwriting properties
		pass
	
	@cached_property
	def config(self):
		logger.info("DefaultKernel used")
		
		import os
		from dotenv import load_dotenv
		load_dotenv()
		
		config = {}
		for name, schema in CONFIG_SCHEMA.items():
			value = os.getenv(name.upper()) or defaults.config.get(name)
			if value is not None and value != "":
				config[name] = schema(value)
		
		logger.debug(f"Loaded {config=}")
		return config
	
	@cached_property
	def adapter(self):
		return TypeAdapter(self.config['retry'])