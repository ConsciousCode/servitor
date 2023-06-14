'''
@module servitor

Semantic functions for LLMs.

Allows the trivial creation of "semantic functions" using decorators, which 
use LLMs as a black box to execute the given task.
'''

from .kernel import Kernel, DefaultKernel
from .adapter import Adapter, TaskAdapter, PlainAdapter, TypeAdapter, ChainOfThoughtAdapter, ChainOfThought
from .complete import Connector, Completion, Delta
from .util import ParseError, ThrottleError, BusyError

semantic = DefaultKernel()