'''
Adapter protocol and implementations. Adapters are used to interface LLM
natural language with programmatic logic, including parsing, error recovery,
and session maintenance.
'''

import inspect
import re
import hjson

from .util import default, logger, typename, typecast, build_signature, build_task
from .defaults import RETRY
from .typings import ABC, typing, override, NamedTuple, TypeAlias, Any, Union, Callable, Generator

class Adapter(ABC):
	'''Adapter protocol. Callable which returns a bidirectional generator returning values.'''
	
	registry: dict[str, type['Adapter']] = {}

	def __init__(self, config):
		super().__init__()
	
	def __call__(self, task: str|Callable, *args, **kwargs) -> Generator[str, str, Any]:
		'''
		Prompt the LLM, parse the response, and fix any mistakes. Should be a
		bidirectional generator:
		Yield: Prompts for the LLM.
		Send: Completions from the LLM.
		Return: Parsed value.
		'''
	
	@staticmethod
	def register(name):
		'''Register an adapter by name.'''
		
		def decorator(cls):
			logger.debug(f"Registering adapter {cls.__name__} as {name!r}")
			Adapter.registry[name] = cls
			return cls
		return decorator
	
	@classmethod
	def find(cls, key: Union[str, 'Adapter']) -> 'Adapter':
		'''Find an adapter by name or return the key if it is already an adapter.'''
		
		if isinstance(key, str):
			return cls.registry[key]
		return key

@Adapter.register("task")
class TaskAdapter(Adapter):
	'''Do-nothing adapter.'''
	
	@override
	def __call__(self, origin, *args, **kwargs):
		logger.debug(f"TaskAdapter(task, *{args!r}, **{kwargs!r})")
		res = yield build_task(origin, args, kwargs)
		return res

@Adapter.register("plain")
class PlainAdapter(Adapter):
	'''Simple minimalist adapter.'''
	
	PROMPT = "Q: {task}\nA:"
	FIX = "{task}{response}\nParsing failed, {error}\nFix all parsing mistakes above:\n"
	
	def __init__(self, config):
		self.retry = config.get("retry", RETRY)
	
	@override
	def __call__(self, origin, *args, **kwargs):
		'''Build a task, prompt the LLM, parse the response, and fix any mistakes.'''
		
		
		task = self.task(origin, args, kwargs)
		prompt = self.prompt(origin, task, args, kwargs)
		logger.debug(f"{type(self).__name__} {prompt=}")
		res = yield prompt
		for i in range(self.retry):
			try:
				logger.debug(f"Parsing response: {res}")
				return self.parse(origin, res)
			except Exception as e:
				logger.debug(f"Failed to parse response: {res}")
				logger.exception(e)
				res = yield self.fix(task, res, e)
	
	def format(self, text, *args, **kwargs):
		'''Adds dedent preprocessing to format.'''
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
		You are to act as a magic interpreter. Given a function description and arguments, provide the best possible answer as a plaintext JSON literal like `return "value"`. Only respond with the answer.
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
				lines.extend(f"\t{k}: " + hjson.dumps(v, indent='\t') for k, v in params.items()),
				lines.append("}")
			return "\n".join(lines)
		
		return '\n'.join([
			"task: " + "origin",
			"args: " + hjson.dumps(args, indent='\t'),
			"kwargs: " + hjson.dumps(kwargs, indent='\t')
		])
	
	def prompt(self, origin, task, args, kwargs):
		'''Add some extra type hints for the LLM if the return type is simple.'''
		
		# Type hints break chat-based models, which think they have to close the
		#  string, list, or dict. Maybe we'll fix it later.
		"""
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
		"""
		return super().prompt(origin, task, args, kwargs)
	
	def parse(self, origin, res):
		'''Parse using HJSON and validate with the return value annotation.'''
		
		# Hack - sometimes LLMs will surround their responses with backticks
		#  because Markdown is a common format that contains JSON
		res = res.strip(" `")
		
		# Hack - sometimes LLMs (especially chat-based ones) will prepend with "return"
		if res.startswith("return "):
			res = res[7:]
		
		# Hack - LLMs struggle to understand tuple return type in JSON format must be a list
		#  so we look for parentheses and replace them with brackets
		if callable(origin):
			ret = inspect.signature(origin).return_annotation
			if typing.get_origin(ret) == tuple:
				if res.startswith("("):
					res = "[" + res[1:]
				if res.endswith(")"):
					res = res[:-1] + "]"
		
		res = hjson.loads(res)
		if callable(origin):
			ret = inspect.signature(origin).return_annotation
			if ret is not inspect.Signature.empty:
				return typecast(res, ret)
		
		return res

class ChainOfThought(NamedTuple):
	'''Result of the ChainOfThoughtAdapter'''
	thoughts: list[str]
	answer: Any

@Adapter.register("chain")
class ChainOfThoughtAdapter(TypeAdapter):
	'''
	Adapter for using Chain of Thought to improve reasoning capabilities.
	'''
	
	PROMPT = """
		You are to act as a magic interpreter. Given a function description and arguments, list your thoughts step by step separated by newlines. When you have a final answer, output `return(answer)` where `answer` is a plaintext JSON literal matching the function signature as the last line. Example:
		def: today(text) -> str
		doc: Get the day of the week from a statement.
		args: {{
			text: The day before two days after the day before tomorrow is Saturday.
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
		
		raise ValueError("No `return(answer)` statement found.")

AdapterProtocol: TypeAlias = Callable[..., Generator[str, str, Any]]
AdapterRef: TypeAlias = str|Adapter|AdapterProtocol