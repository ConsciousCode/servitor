'''
Adapter protocol and implementations. Adapters are used to interface LLM
natural language with programmatic logic, including parsing, error recovery,
and session maintenance.
'''

from typing import Protocol, Callable, Generator, Any, NamedTuple, Optional, get_origin, get_args, get_type_hints, Union, Literal
import types
import inspect
import re
import hjson

from .util import Registry, default, build_task

def typename(cls) -> str:
	'''Convert an annotation into a typename string for the LLM.'''
	
	origin = get_origin(cls)
	if isinstance(cls, types.GenericAlias):
		return str(cls)
	elif origin is Union:
		args = get_args(cls)
		if types.NoneType in args:
			rest = [typename(t) for t in args if t is not types.NoneType]
			base = rest[0] if len(rest) == 1 else f'({"|".join(rest)})'
			return f"{base}?"
		return "|".join(typename(t) for t in args)
	elif origin is Literal:
		return repr(get_args(cls)[0])
	elif cls is Any:
		return "any"
	elif hints := get_type_hints(cls):
		fields = []
		for field, ft in hints.items():
			fields.append(field if ft is Any else f"{field}: {typename(ft)}")
		return f"{{{', '.join(fields)}}}"
	elif isinstance(cls, type):
		return cls.__name__
	elif isinstance(cls, str):
		return cls
	elif cls is ...:
		return "..."
	else:
		return repr(cls)

def build_signature(origin: Callable):
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
	
	return fndef

class Adapter(Protocol):
	'''Adapter protocol. Callable which returns a bidirectional generator returning values.'''
	
	registry = Registry()
	
	def __call__(self, origin: str|Callable, *args, **kwargs) -> Generator[str, str, Any]:
		'''Build a task, prompt the LLM, parse the response, and fix any mistakes.'''
		pass
	
	@staticmethod
	def register(name):
		def decorator(cls):
			cls.adapter_name = name
			Adapter.registry.register(name, cls.supports)
			return cls
		return decorator
	
	@classmethod
	def supports(cls, key):
		if key == cls.adapter_name:
			return cls

@Adapter.register("task")
class TaskAdapter(Adapter):
	'''Do-nothing adapter.'''
	
	def __call__(self, origin, *args, **kwargs):
		res = yield build_task(origin, args, kwargs)
		return res

@Adapter.register("plain")
class PlainAdapter(Adapter):
	'''Simple minimalist adapter.'''
	
	PROMPT = "{task}\nAnswer:"
	FIX = "{task}{response}\nParsing failed, {error}\nFix all parsing mistakes above:\n"
	
	def __init__(self, retry=None):
		self.retry = default(retry, 3)
	
	def __call__(self, origin, *args, **kwargs):
		'''Build a task, prompt the LLM, parse the response, and fix any mistakes.'''
		
		task = self.task(origin, args, kwargs)
		res = yield self.prompt(task, args, kwargs)
		for _ in range(self.retry):
			try:
				return self.parse(origin, res)
			except Exception as e:
				res = yield self.fix(task, res, e)
	
	def format(self, text, *args, **kwargs):
		'''Adds dedent preprocessing to format.'''
		
		if m := re.match(r"""^(\s+)|^\S.*\n(\s+)""", text):
			text = re.sub(rf"^{m[m.lastindex]}", "", text, flags=re.M)
		
		return text.format(*args, **kwargs)
	
	def task(self, origin, args, kwargs):
		'''Build a task from an origin for the LLM.'''
		return build_task(origin, args, kwargs)
	
	def prompt(self, task, args, kwargs):
		'''Build a prompt for the LLM to give context to the task.'''
		return self.format(self.PROMPT, task=task, args=args, kwargs=kwargs)
	
	def parse(self, origin, res):
		'''Parse the response from the LLM.'''
		return res
	
	def fix(self, task, res, error):
		'''Build a fix prompt for the LLM to correct parsing mistakes.'''
		return self.format(self.FIX, task=task, response=res, error=error)

@Adapter.register("type")
class TypeAdapter(PlainAdapter):
	'''
	Uses HJSON to use less tokens and account for common mistakes made by the LLM.
	'''
	
	# Main prompt for the semantic kernel. Carefully crafted, models seem to need
	#  the example to have a string result to make sure they quote strings, all
	#  other types don't have any issue.
	# Possible lead: asking for result in a JSON list even if it's only one item
	PROMPT = """
		You are to act as a magic interpreter. Given a function description and arguments, provide the best possible answer as a plaintext JSON literal.
		Example:
		{{
			def: object_of(text) -> str
			doc: Get the grammatical object in a statement.
			args:
			{{
				text: I never said that!
			}}
		}}
		return "that"
		{task}
		return
	""".strip()
	
	def task(self, origin, args, kwargs):
		'''Builds the task as a dictionary.'''
		
		if callable(origin):
			prompt = {
				"def": build_signature(origin),
				"doc": build_task(origin, args, kwargs),
				"args": inspect.getcallargs(origin, *args, **kwargs)
			}
		else:
			prompt = {
				"task": origin,
				"args": args,
				"kwargs": kwargs
			}
		
		return hjson.dumps(prompt, indent='\t')
	
	def parse(self, origin, res):
		'''Parse using HJSON and validate with the return value annotation.'''
		
		res = hjson.loads(res)
		if callable(origin):
			ret = inspect.signature(origin).return_annotation
			if isinstance(ret, type):
				return ret(res)
		
		return res

class ChainOfThought(NamedTuple):
	thoughts: list[str]
	answer: Any

@Adapter.register("cot")
class ChainOfThoughtAdapter(TypeAdapter):
	'''
	Adapter for using Chain of Thought to improve reasoning capabilities.
	'''
	
	PROMPT = """
		You are to act as a magic interpreter. Given a function description and arguments, list your thoughts step by step separated by newlines. When you have a final answer, output `return(answer)` where `answer` is a plaintext JSON literal matching the function signature as the last line.
		Example:
		{{
			def: today(text) -> str
			doc: Get the day of the week from a statement.
			args:
			{{
				text: The day before two days after the day before tomorrow is Saturday.
			}}
		}}
		Thoughts:
		The day before tomorrow is today. 
		Two days after that is the day after tomorrow.
		The day before that is tomorrow.
		Tomorrow is Saturday, so today is Friday.
		return("Friday")
		
		{task}
	""".split()
	
	def parse(self, origin, res):
		*thoughts, answer = res.splitlines()
		
		# Greedy match to the last parenthesis in the string to avoid having
		#  to parse it, with optional whitespace just in case.
		if m := re.search(r"return[\s\n]*\([\s\n]*(.+)[\s\n]*\)[^\)\n]*$", answer):
			return ChainOfThought(thoughts, super().parse(origin, m[1]))
		
		raise ValueError("No return statement found.")