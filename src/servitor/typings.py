'''
Miscellaneous types and type definitions. Also provides a common point to
import types from the correct standard libraries, since it's a bit of a mess
especially between typing and collections.abc, with many typing bindings
being deprecated.
'''

import typing
from typing import TypeVar, TypeAlias, GenericAlias, Optional, Union, Literal, Any, TypedDict, Protocol, NamedTuple
from collections import UserDict
# collections.abc versions are canonical, typing versions are deprecated
from collections.abc import *
from abc import ABC, abstractmethod
from types import NoneType
from enum import Enum

NotRequired: TypeAlias = getattr(typing, "NotRequired", Optional)
override = getattr(typing, "override", lambda x: x)

class SyncDocstringOrigin(Protocol):
	'''Semantic function built using its docstring.'''
	__doc__: str
	def __call__(self, *args, **kwargs) -> NoneType: ...
SyncSemanticOrigin: TypeAlias = str|SyncDocstringOrigin|Callable[..., str]
'''A synchronous semantic function origin.'''

class AsyncDocstringOrigin(Protocol):
	'''Semantic function built using its docstring.'''
	__doc__: str
	def __call__(self, *args, **kwargs) -> Awaitable[NoneType]: ...
AsyncSemanticOrigin: TypeAlias = AsyncDocstringOrigin|Callable[..., Awaitable[str]]
'''An asynchronous semantic function origin.'''

SemanticOrigin: TypeAlias = SyncSemanticOrigin|AsyncSemanticOrigin
'''Any origin for a semantic function.'''

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
