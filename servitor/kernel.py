'''
Kernels aggregate the machinery and defaults needed to create, coordinate, and
execute semantic functions. Unlike MS Semantic Kernels, they do not store any
semantic functions themselves to be exposed to a single LLM instance to use as
tools, though this sort of functionality can be built *on top* of kernels.
'''

from functools import wraps, cached_property
from typing import Optional, Callable
import inspect
import asyncio

from .util import default, logger
from .adapter import Adapter, TypeAdapter
from .complete import Connector, Completion

from .util import build_task

def optargs_decorator_method(fn_decorator):
	'''Create a method decorator which can optionally accept arguments.'''
	
	@wraps(fn_decorator)
	def optional_arg(self, fn_decoratee=None, *args, **kwargs):
		'''Generic optional-argument decorator.'''
		
		@wraps(fn_decorator)
		def forward(fn_decoratee):
			'''Forward all arguments to the decorator.'''
			return fn_decorator(self, fn_decoratee, *args, **kwargs)
		
		if fn_decoratee is not None:
			return forward(fn_decoratee)
		return forward
	return optional_arg

def wrap_synchrony(origin, fn):
	'''Wrap fn (async) to have the same synchrony as origin.'''
	
	if inspect.iscoroutinefunction(origin):
		return fn
	
	@wraps(fn)
	def await_fn(*args, **kwargs):
		return asyncio.run(fn(*args, **kwargs))
	return await_fn

def force_sync(fn):
	'''
	Given an async function, wrap it as a synchronous function. This assumes
	the function is "essentially" synchronous. It will throw an error if it
	tries to await anything!
	'''
	
	# No need to wrap
	if not inspect.iscoroutinefunction(fn):
		return fn
	
	@wraps(fn)
	def sync_fn(*args, **kwargs):
		coro = fn(*args, **kwargs).__await__()
		try:
			coro = coro.send(None)
			if coro is not None:
				raise RuntimeError("Semantic function definitions cannot await!")
		except StopIteration as e:
			return e.value
		raise RuntimeError("Malformed awaitable did not stop iteration.")
	return sync_fn

class Kernel:
	'''
	Aggregates machinery needed to execute and coordinate semantic functions.
	'''
	
	def __init__(self, connector: str|Connector, adapter: str|Adapter, config=None, **kwargs):
		'''
		Parameters:
			connector: Connector for LLM instance
			adapter: Parsing and session maintenance for the LLM's responses
			config: Default configuration for connector (if unset, kwargs are used)
			
		'''
		self.connector = Connector.registry.find(connector)
		self.adapter = Adapter.registry.find(adapter)
		self.config = default(config, kwargs)
	
	def complete(self, origin: str|Callable, *args, config=None, **kwargs) -> Completion:
		'''Normal completion.'''
		
		return self.connector(build_task(origin, args, kwargs), **default(config, kwargs))
	
	@optargs_decorator_method
	def __call__(self,
			origin: str|type|Callable,
			adapter: Optional[Callable]=None,
			*,
			config=None,
			**kwargs
		):
		'''
		Build a semantic function from its metadata. Also works as a decorator.
		
		Parameters:
			task: Task description, either a string or a callable that returns
			  a string or a dictionary.
			adapter: Adapter to use for this function.
		'''
		
		adapter = Adapter.registry.find(default(adapter, lambda: self.adapter))
		config = default(config, kwargs)
		sync_origin = force_sync(origin)
		
		name = ("async " if inspect.iscoroutinefunction(origin) else "") + origin.__name__
		logger.debug(f"Building semantic function {name} with adapter {adapter.adapter_name} and {config=}")
		
		async def semantic_fn(*args, **kwargs):
			'''Semantic function closure.'''
			
			conf = {**self.config, **config}
			try:
				task = adapter(sync_origin, *args, **kwargs)
				prompt = next(task)
				
				while prompt:
					prompt = task.send(await self.connector(prompt, **conf))
			except StopIteration as e:
				return e.value
		
		# Wrap the __call__ method if it's a class
		if isinstance(origin, type):
			if not hasattr(origin, "__call__"):
				raise TypeError(f"Task {origin} instances must be callable.")
			
			origin_class = origin
			origin = origin.__call__ # Assign so semantic_fn can see it
			semantic_fn = wrap_synchrony(origin, semantic_fn)
			origin_class.__call__ = wraps(origin)(semantic_fn)
			origin_class.__name__ = f"{origin_class.__name__}.__call__"
			return origin_class
		
		if callable(origin):
			semantic_fn = wraps(origin)(semantic_fn)
		return wrap_synchrony(origin, semantic_fn)

class DefaultKernel(Kernel):
	'''Kernel with reasonable defaults for quick use. Does nothing until used.'''
	
	def __init__(self):
		# no super().__init__() to avoid overwriting properties
		pass
	
	@cached_property
	def config(self):
		import os
		from dotenv import load_dotenv
		load_dotenv()
		
		if api_key := os.getenv("OPENAI_API_KEY"):
			default_model = "gpt-3.5-turbo"
			
			provider = "openai"
			organization = os.getenv("OPENAI_ORGANIZATION")
		else:
			raise RuntimeError("No LLM provider configured.")
		
		config = dict(
			temperature=float(os.getenv("TEMPERATURE", 0)),
			max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
			top_p=float(os.getenv("TOP_P", 0.9)),
			frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", 0)),
			presence_penalty=float(os.getenv("PRESENCE_PENALTY", 0)),
			best_of=int(os.getenv("BEST_OF", 1)),
			max_retry=int(os.getenv("MAX_RETRY", 3)),
			max_concurrent=int(os.getenv("MAX_CONCURRENT", 1)),
			max_rate=int(os.getenv("MAX_RATE", 1)),
			max_period=float(os.getenv("MAX_PERIOD", 60)),
			model=os.getenv("MODEL", default_model),
			api_key=api_key,
			provider=provider,
			organization=organization
		)
		logger.debug(f"Loaded configuration: {config}")
		return config
	
	@cached_property
	def connector(self):
		match self.config['provider']:
			case "openai":
				from .openai import OpenAIConnector
				return OpenAIConnector(
					self.config["api_key"],
					self.config["model"],
					self.config["max_concurrent"],
					self.config['max_rate'],
					self.config['max_period']
				)
			case _:
				raise RuntimeError(f"Unknown LLM provider {self.config['provider']}.")
	
	@cached_property
	def adapter(self):
		return TypeAdapter(self.config['max_retry'])