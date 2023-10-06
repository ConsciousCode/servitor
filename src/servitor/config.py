from .typings import UserDict, KernelConfig
from .util import parse_bool

class DefaultConfig(UserDict):
	'''Config which defers to a given default.'''

	def __init__(self, config, default: KernelConfig):
		super().__init__(config)
		self._default = default
	
	def __iter__(self):
		yield from super()
		yield from self._default
	
	def __contains__(self, name):
		return name in super() or name in self._default
	
	def __len__(self):
		return len(super()) + len(self._default)
	
	def __str__(self):
		s = ', '.join(f"{k}: {v}" for k, v in self.items())
		return f"{{{s}}}"
	
	def __eq__(self, other):
		for k, v in self.items():
			if other.get(k, None) != v:
				return False
		return True
	
	def __ne__(self, other):
		return not (self == other)
	
	def __reversed__(self):
		yield from reversed(self._default)
		yield from reversed(super())
	
	def __missing__(self, name):
		return self._default[name]
	
	def __repr__(self):
		return f"DefaultConfig({super()!r}, {self._default!r})"
	
	def keys(self):
		yield from super().keys()
		yield from self._default.keys()
	
	def values(self):
		yield from super().values()
		yield from self._default.values()
	
	def items(self):
		yield from super().items()
		yield from self._default.items()
	
	def copy(self):
		return DefaultConfig(self, self._default)

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
'''Configuration schema for env vars.'''