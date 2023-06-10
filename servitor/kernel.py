'''
Kernels aggregate the machinery and defaults needed to create, coordinate, and
execute semantic functions. Unlike MS Semantic Kernels, they do not store any
semantic functions themselves to be exposed to a single LLM instance to use as
tools, though this sort of functionality can be built *on top* of kernels.
'''

from functools import wraps, cached_property
from typing import Optional, Callable, Any

from .util import default
from .adapter import Adapter, TypeAdapter
from .complete import Connector, Completion

from .util import build_task

def decorator(fn_decorator):
	'''Create a decorator which can optionally accept arguments.'''
	
	@wraps(fn_decorator)
	def optional_arg(fn_decoratee=None, *args, **kwargs):
		'''Generic optional-argument decorator.'''
		
		@wraps(fn_decorator)
		def forward(fn_decoratee):
			'''Forward all arguments to the decorator.'''
			return fn_decorator(fn_decoratee, *args, **kwargs)
		
		return forward if fn_decoratee is None else forward(fn_decoratee)
	return optional_arg

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
	
	@decorator
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
		
		adapter = default(adapter, lambda: self.adapter)
		config = default(config, kwargs)
		
		async def semantic_fn(*args, **kwargs):
			'''Semantic function closure.'''
			
			conf = {**self.config, **config}
			try:
				task = adapter(origin, *args, **kwargs)
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
			origin_class.__call__ = wraps(origin)(semantic_fn)
			return origin_class
		
		if callable(origin):
			semantic_fn = wraps(origin)(semantic_fn)
		return semantic_fn

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
		
		return dict(
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