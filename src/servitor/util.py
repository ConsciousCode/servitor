'''
Common utilities.
'''

import dataclasses
import logging
import os
from functools import wraps
import inspect

logger = logging.getLogger("servitor")
if loglevel := os.getenv("LOG_LEVEL"):
	loglevel = loglevel.upper()
	logger.addHandler(logging.StreamHandler())
	logger.setLevel(loglevel)
	logger.info(f"Set log level to {loglevel}")

# Make sure relative imports are after logging
from .typings import *

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

def async_await(fn):
	'''Decorator for converting an async method into a generator function like __await__'''
	@wraps(fn)
	def wrapper(*args, **kwargs):
		return fn(*args, **kwargs).__await__()
	return wrapper

def parse_bool(value: Any) -> bool:
	'''Permissive parsing of boolean values.'''
	
	if isinstance(value, str):
		value = value.lower()
	
	match value:
		case 1 | True | "t" | "true" | "y" | "yes" | "on" | "enable" | "1":
			return True
		
		case 0 | False | None | "f" | "false" | "n" | "no" | "off" | "disable" | "0" | "":
			return False
	
	raise ValueError(f"Cannot convert {value!r} to bool")

def is_TypedDict(cls):
	return (
		issubclass(cls, dict) and
		hasattr(cls, "__required_keys__") and
		hasattr(cls, "__optional_keys__")
	)

def typename(cls) -> str:
	'''Convert an annotation into a typename string for the LLM.'''
	
	cls = normalize_type(cls)
	
	origin, args = typing.get_origin(cls), typing.get_args(cls)
	if origin is Union:
		if NoneType in args:
			rest = [typename(t) for t in args if t is not NoneType]
			base = rest[0] if len(rest) == 1 else f'({"|".join(rest)})'
			return f"{base}?"
		return "|".join(typename(t) for t in args)
	elif origin is Literal:
		return "|".join(map(repr, args))
	elif cls is Any:
		return "any"
	elif cls is str:
		return "string"
	elif hints := typing.get_type_hints(cls):
		fields = []
		for field, ft in hints.items():
			fields.append(field if ft is Any else f"{field}: {typename(ft)}")
		if not cls.__total__:
			fields.append("...")
		return f"{{{', '.join(fields)}}}"
	elif isinstance(cls, GenericAlias):
		return repr(cls)
	elif is_TypedDict(cls):
		fields = []
		hints = typing.get_type_hints(cls)
		for hint in hints:
			c = "?" if hint in cls.__optional_keys__ else ""
			fields.append(f"{hint}{c}: {typename(hints[hint])}")
		
		return f"{cls.__name__}{{{', '.join(fields)}}}"
	elif isinstance(cls, type):
		return cls.__name__
	elif isinstance(cls, str):
		return cls
	elif cls is ...:
		return "..."
	else:
		return repr(cls)

def build_generic(origin, args):
	'''Indirection helper for building generic types without IDE complaining.'''
	return origin[tuple(map(normalize_type, args))]

def normalize_type(cls) -> type:
	'''Normalize the type annotation for use in typechecking.'''
	
	if isinstance(cls, str):
		return cls
	
	if cls in {object, Enum, Union, Any, bool, int, float}: return cls
	if cls in {None, NoneType}: return NoneType
	# Sequence types
	if cls in {str, typing.Text}: return str
	if cls in {list, typing.List}: return list
	if cls in {tuple, typing.Tuple}: return tuple
	# Set types
	if cls in {set, Set, typing.Set, MutableSet, typing.MutableSet}: return set
	if cls in {frozenset, typing.FrozenSet}: return frozenset
	# dict is the only builtin mapping type
	if cls in {dict, typing.Dict, Mapping, typing.Mapping, MutableMapping, typing.MutableMapping}: return dict
	
	# Abstract types
	if cls in {Sequence, typing.Sequence, MutableSequence, typing.MutableSequence}:
		return Sequence
	if cls in {Iterable, typing.Iterable}:
		return Iterable
	
	# Generics
	
	origin, args = typing.get_origin(cls), typing.get_args(cls)
	
	# Optional is an alias for Union[None, T] and Optional[Union[...]] == Union[..., None]
	if origin is Union:
		return build_generic(Union, args)
	
	return build_generic(normalize_type(origin), args)

def typecast(value: Any, target: str|type|None) -> Any:
	'''Reasonable typecasting and typechecking for LLM values.'''
	
	target = normalize_type(target)
	
	# String annotations aren't qualified
	if isinstance(target, str):
		return value
	
	# Simple casts
	if type(value) is target or target is Any: return value
	if target is float: return float(value)
	if target is str: return str(value)
	
	# Unqualified collection types
	if target is tuple: return tuple(value)
	if target is list: return list(value)
	if target is dict: return dict(value)
	
	# Leaf casts that need some handling
	if target is NoneType:
		if value is None:
			return value
		raise TypeError(f"Cannot convert {typename(type(value))} to NoneType")
	
	if target is bool:
		# May be a misunderstanding by the LLM of the format of a bool
		if isinstance(value, str):
			return parse_bool(value)
		
		# Probably indicates an error in the LLM
		if isinstance(value, (tuple, list, set, dict)):
			raise TypeError(f"Cannot convert {type(value).__name__} to bool")
		
		# Otherwise, just try to convert it
		return bool(value)
	
	if target is int:
		if isinstance(value, str):
			return int(value, 0)
		# Non-string can't have base 0, already throws on bad types
		return int(value)
	
	# NamedTuple / dataclass
	
	if issubclass(target, tuple) and hasattr(target, "_fields"):
		# Typed
		if notes := inspect.get_annotations(target):
			fields = {k: typecast(v, t) for (k, t), v in zip(notes.items(), value)}
		# Untyped
		else:
			fields = dict(zip(target._fields, value))
		
		return target(**fields)
	
	if dataclasses.is_dataclass(target):
		fields = {f.name: f.type for f in dataclasses.fields(target)}
		return target(**{k: typecast(v, fields[k]) for k, v in value.items()})
	
	# Generic types
	
	origin, args = typing.get_origin(target), typing.get_args(target)
	
	if origin == Literal:
		if value in args:
			return value
		raise ValueError(f"{value!r} is not {target}")
	
	if origin == Union:
		for arg in args:
			try:
				return typecast(value, arg)
			except Exception:
				continue
		
		raise TypeError(f"Cannot convert {value!r} to {typename(target)}")
	
	if origin == Mapping:
		if isinstance(value, Mapping):
			return value
		raise TypeError(f"Cannot convert {value!r} to {typename(target)}")
	
	if origin == Sequence:
		if isinstance(value, Sequence):
			return value
		raise TypeError(f"Cannot convert {value!r} to {typename(target)}")
	
	# Recursive conversions for typed collections
	
	if origin in {tuple, list, set, frozenset}:
		return origin(typecast(v, args[0]) for v in value)
	
	if origin is dict:
		# Bug? dict args can be any length
		if len(args) != 2:
			return dict(value)
		kt, vt = args
		return {typecast(k, kt): typecast(v, vt) for k, v in value.items()}
	
	# Last ditch effort
	return target(value)

def build_signature(origin: Callable) -> str:
	'''Build a typed function signature for a given function.'''
	
	sig = inspect.signature(origin)
	params = []
	for name, param in sig.parameters.items():
		if param.annotation is inspect.Parameter.empty:
			params.append(name)
		else:
			params.append(f"{name}: {typename(param.annotation)}")
	
	fndef = f"{origin.__name__ or ''}({', '.join(params)})"
	if sig.return_annotation is not inspect.Signature.empty:
		fndef += f" -> {typename(sig.return_annotation)}"
	inspect.getsource
	return fndef