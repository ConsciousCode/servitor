'''
Adapter protocol and implementations. Adapters are used to interface LLM
natural language with programmatic logic, including parsing, error recovery,
and session maintenance.
'''

from typing import Protocol, Callable, Generator, NamedTuple, Any, get_origin, get_args, get_type_hints, Union, Literal, Mapping
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
			return cls()

@Adapter.register("task")
class TaskAdapter(Adapter):
	'''Do-nothing adapter.'''
	
	def __call__(self, origin, *args, **kwargs):
		res = yield build_task(origin, args, kwargs)
		return res

@Adapter.register("plain")
class PlainAdapter(Adapter):
	'''Simple minimalist adapter.'''
	
	PROMPT = "Q: {task}\nA:"
	FIX = "{task}{response}\nParsing failed, {error}\nFix all parsing mistakes above:\n"
	
	def __init__(self, retry=None):
		self.retry = default(retry, 3)
	
	def __call__(self, origin, *args, **kwargs):
		'''Build a task, prompt the LLM, parse the response, and fix any mistakes.'''
		
		task = self.task(origin, args, kwargs)
		#print("TASK", origin.__name__, type(task), task)
		res = yield self.prompt(origin, task, args, kwargs)
		for _ in range(self.retry):
			try:
				return self.parse(origin, res)
			except Exception as e:
				res = yield self.fix(task, res, e)
	
	def format(self, text, *args, **kwargs):
		'''Adds dedent preprocessing to format.'''
		print("PlainAdapter.format", text)
		if m := re.match(r"""^(\s+)|^\S.*\n(\s+)""", text):
			text = re.sub(rf"^{m[m.lastindex]}", "", text, flags=re.M)
		
		return text.format(*args, **kwargs)
	
	def task(self, origin, args, kwargs):
		'''Build a task from an origin for the LLM.'''
		return build_task(origin, args, kwargs)
	
	def prompt(self, origin, task, args, kwargs):
		'''Build a prompt for the LLM to give context to the task.'''
		return self.format(self.PROMPT, task=task, args=args, kwargs=kwargs)
	
	def parse(self, origin, res):
		'''Parse the response from the LLM.'''
		return res
	
	def fix(self, task, res, error):
		'''Build a fix prompt for the LLM to correct parsing mistakes.'''
		return self.format(self.FIX, task=task, response=res, error=error)

def indent(text, amount, ch='\t'):
	'''Indent a block of text.'''
	pre = ch * amount
	return pre + re.sub("^", pre, text, flags=re.M)

def escape(text):
	return text.replace("\\", "\\\\")

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
		You are to act as a magic interpreter. Given a function description and arguments, provide the best possible answer as a plaintext JSON literal like `return "value"`.
		{task}
		return
	""".strip()
	
	def task(self, origin, args, kwargs):
		if callable(origin):
			params = inspect.getcallargs(origin, *args, **kwargs)
			lines = [
				f"def: {build_signature(origin)}",
				f"doc: {build_task(origin, args, kwargs)}"
			]
			if len(params):
				lines.append("args: {")
				lines.extend(f"\t{k}: {hjson.dumps(v)}" for k, v in params.items()),
				lines.append("}")
			return "\n".join(lines)
		return f"task: {origin}\nargs: {hjson.dumps(args)}\nkwargs: {hjson.dumps(kwargs)}"
	
	def prompt(self, origin, task, args, kwargs):
		'''Add some extra type hints for the LLM if the return type is simple.'''
		if callable(origin):
			ret = inspect.signature(origin).return_annotation
			base = get_origin(ret) or ret
			if ret == str:
				task += ' "'
			elif base == list or base == tuple:
				task += ' ['
			elif base == dict or base == Mapping:
				task += ' {'
			elif base == Union:
				if all(get_origin(x) == Literal and get_args(x) == (str,) for x in get_args(ret)):
					task += ' "'
		return super().prompt(origin, task, args, kwargs)
	
	def parse(self, origin, res):
		'''Parse using HJSON and validate with the return value annotation.'''
		
		res = hjson.loads(res)
		if callable(origin):
			ret = inspect.signature(origin).return_annotation
			if isinstance(ret, type):
				return ret(res)
		
		return res

class ChainOfThought(NamedTuple):
	'''Result of the ChainOfThoughtAdapter'''
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
	""".strip()
	
	def parse(self, origin, res):
		*thoughts, answer = res.splitlines()
		
		# Greedy match to the last parenthesis in the string to avoid having
		#  to parse it, with optional whitespace just in case.
		if m := re.search(r"return[\s\n]*\([\s\n]*(.+)[\s\n]*\)[^\)\n]*$", answer):
			return ChainOfThought(thoughts, super().parse(origin, m[1]))
		
		raise ValueError("No `return(value)` statement found.")