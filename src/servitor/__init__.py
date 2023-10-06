'''
@module servitor

Semantic functions for LLMs.

Allows the trivial creation of "semantic functions" using decorators, which 
use LLMs as a black box to execute the given task.
'''

# Import util first so logging is set before everything else
from .util import ParseError, ThrottleError, BusyError

from .typings import SyncSemanticOrigin, AsyncSemanticOrigin, SemanticOrigin 
from .kernel import SemanticFunction, Kernel, DefaultKernel
from .adapter import Adapter, TaskAdapter, PlainAdapter, TypeAdapter, ChainOfThoughtAdapter, ChainOfThought
from .connectors import Connector, Completion

semantic = DefaultKernel()
semantic.__name__ = "semantic"
semantic.__qualname__ = f"{__name__}.semantic"