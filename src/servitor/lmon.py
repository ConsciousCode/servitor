'''
Language Model Object Notation (LMON) is a permissive superset of JSON using
typed schemas and with an emphasis on streaming. All values are implicitly
strings or aggregates, with actual parsing deferred to the user so it can be
driven by the schema.

Inspirations:
* ijson
* HJSON
* Strict YAML (https://hitchdev.com/strictyaml/features-removed/ - no implicit typing)
* demjson - permissive JSON parser
'''
import re
from functools import wraps
import typing
from typing import Literal, NamedTuple, Optional, Union, Iterable, NoneType, Any
import types
import unicodedata
from decimal import Decimal
from dataclasses import dataclass

def coroutine(fn):
	'''
	Wraps a generator which is intended to be used as a pure coroutine by
	.send()ing it values. The only thing that the wrapper does is calling
	.next() for the first time which is required by Python generator protocol.
	'''
	@wraps(fn)
	def wrapper(*args, **kwargs):
		g = fn(*args, **kwargs)
		next(g)
		return g
	return wrapper

LEXEMERE = re.compile(r"""
	# Empty string
	((?:''|""|``)(?!$)|(?:'{6}|"{6}|`{6}))|
	# String with content
	('''|\"""|```|['"`](?!$))|
	# Unquoted string
	([^\n:,\(\)\[\]\{\}]+)|
	# Sequence start
	([\(\[\{])|
	# Sequence end
	([\)\]\}])|
	# Other
	(\S)
""", re.X)
UNQUOTED_RE = re.compile(".*[^,]+")

class Lexemere(NamedTuple):
	'''
	Lexeme + -mere, a part of a lexeme. Originally SubLexeme, but then I saw the regex
	name LEXEME_RE and "lexemere" just makes so much sense.
	'''
	kind: Literal["eof", "str", ":", ",", '"', "(", ")", "[", "]", "{", "}"]
	value: str

def coro_next():
	try:
		buf = yield
	except GeneratorExit:
		buf = ""
	return buf

@coroutine
def lexer(target):
	buf = yield from coro_next()
	
	while True:
		# Consume more data
		if not (m := LEXEMERE.match(buf)):
			buf += yield from coro_next()
			if buf == "": # EOF
				target.send(Lexemere("eof", "eof"))
				break
			
		# Empty string
		elif q := m[1]:
			q = q[len(q)//2:]
			target.send(Lexemere('str_start', q))
			target.send(Lexemere('str_end', q))
			buf = buf[m.end():]
		
		# Quoted string
		elif q := m[2]:
			STREND_RE = re.compile(rf"((?:\\.|(?!{q})[^\\]+?)*){q}")
			
			buf = buf[m.end():]
			target.send(Lexemere('str_start', q))
			
			# Consume until we see the quote
			while not (m := STREND_RE.match(buf)):
				buf = yield from coro_next()
				target.send(Lexemere('str', buf))
			
			# Consume anything before the quote
			if m[1]:
				target.send(Lexemere('str', m[1]))
			target.send(Lexemere('str_end', q))
		
		# Unquoted string
		elif m[3]:
			# Unquoted strings don't send str_start or str_end
			target.send(Lexemere('str', m[3]))
			
			# Special character follows, unquoted string ends
			if not (buf := buf[m.end():]):
				while True:
					buf = yield from coro_next()
					if m := UNQUOTED_RE.match(buf):
						target.send(Lexemere('str', m[0]))
						# Consumed the whole buffer, unquoted string may continue
						if not (buf := buf[m.end():]):
							continue
					break
		
		# Sequence start
		elif m[4]:
			target.send(Lexemere("seq_start", m[4]))
			buf = buf[m.end():]
		
		# Sequence end
		elif m[5]:
			target.send(Lexemere("seq_end", m[5]))
			buf = buf[m.end():]
		
		# Punctuators
		else:
			c, *buf = buf
			target.send(Lexemere(c, c))

STRESC = re.compile(r"""(?x)
	\\(?:
		(["\\/bfnrt])|
		([0-7]{1,3})|
		x([a-fA-F\d]{2})|
		u([a-fA-F\d]{4})|
		U([a-fA-F\d]{8})|
		[uU]\{[a-fA-F\d]+)\}|
		N\{([\w\d]+)\}|
	)
""")
ESCAPES = {
	"\\": "\\",
	"a": "\a", "b": "\b", "e": "\x1b"
	"f": "\f", "v": "\v",
	"n": "\n", "r": "\r", "t": "\t",
}
def _translate(m):
	if esc := m[1]:
		return ESCAPES[esc]
	if esc := m[2]:
		return chr(int(esc, 8))
	if esc := (m[3] or m[4] or m[5] or m[6]):
		return chr(int(esc, 16))
	return unicodedata.lookup(m[7])

def escape(q, text: str):
	return STRESC.sub(_translate, text.replace(f"\\{q}", q))

VALUE_RE = re.compile(r"""(?xi)
	([-+]?(?:(?:\d+\.\d*|\d*\.\d+)(?:e[-+]?\d+)?)|inf(?:inity)?|nan)
	([-+]?0o?[-,_'1-7]+)|
	([-+]?0b[-,_'01]+)
	([-+]?(?:0x|\#)[-,_'a-f\d]+)|
	([-+]?\d+)|
	([tyax]|true|yes|ok|sure|enable|accept|allow)|
	([fndo]|false|no|nope|nah|disable|reject|disallow)|
	(null?|none|nil|void|nothing|undef(?:ined)|invalid)
""")
			  
@dataclass
class Atom:
	'''Atom from a LMON data.'''
	
	quote: Optional[str]
	text: str
	
	def value(self):
		if self.quote == "":
			m = VALUE_RE.match(self.text)
			if m is None:
				return escape(self.text.strip())
			
			match m.lastgroup:
				case "int": return self.int()
				case "float": return self.float()
				case "true": return True
				case "false": return False
				case "null": return None
		
		return escape(self.quote[0], self.text)
	
	def int(self):
		return int(self.text.strip(), 0)
	
	def float(self):
		return float(self.text.strip())
	
	def decimal(self):
		return Decimal(self.text.strip())
	
	def bool(self):
		text = self.text.strip().lower()
		if text in ("t", "true", "y", "yes"):
			return True
		elif text in ("f", "false", "n", "no"):
			return False
		else:
			raise ValueError(f"Invalid boolean value {self.text!r}")
	
	def str(self):
		return self.text

'''
Rules:
* Unquoted strings are unconditionally broken by newlines
* A schema which supports str but not list, tuple, set, or dict will ignore ()[]{} and treat them as strings
* Unquoted strings in a non-dict schema also ignore :

* Quoted strings are not broken by newlines

Need to abstract this somewhat, right now it's a loose intuition of disambiguation rules...

sensitive_to(base, sub, c) -> bool

sensitive_to(None, str, ":") -> False
sensitive_to(list, str, ",") -> True

These sensitivity rules are only relevant for unquoted strings with a str schema

Where these rules get potentially problematic is when we incorporate unions:

list[str]|list[int|list]

bottom <: ... <: top
null <: bool <: int <: float <: str ... <: Any
tuple <: namedtuple <: dataclass

dict[K, V] <: dict[S, T] iff K <: S and V <: T
list[T] <: list[S] iff T <: S
T <: S|U iff T <: S or T <: U
T|U <: S iff T <: S and U <: S

We have two types at any given point:
* The schema type
* The value type (the realized type of the value)

At any point the value type must be a subtype of the schema type - coercions are allowed, but if it can't be done then it's an error
Parse according to the most restrictive schema type we can. So the sensitive_to function can be implemented as a subtype check. list[str] 

container type / value type

list[int] -> list / type[int]
list[int|str] -> list / type[int|str]
list[int]|list[str] -> list / type[int]|type[str]

When encountering "(":
* tuple in schema: parse as tuple
* str in schema: parse as str
* list in schema: parse as list
* set in schema: parse as set
* dict in schema: parse as dict
* error

tuple > str > list > set > dict > error

When encountering "[":
* list in schema: parse as list
* tuple in schema: parse as tuple
* set in schema: parse as set
* dict in schema: parse as dict
* str in schema: parse as str
* error

list > tuple > set > dict > str > error

When enconutering "{":
* dict in schema: parse as dict
* set in schema: parse as list
* list in schema: parse as list
* tuple in schema: parse as tuple
* str in schema: parse as str
* error

dict > set > list > tuple > str > error
'''

class ParseState(NamedTuple):
	state: Literal['value', 'array', 'object']
	value: Any
	schema: type

def generic_args(t: type):
	'''Return the (possibly implicit) arguments of a generic.'''
	
	if origin := typing.get_origin(t):
		return typing.get_args(t) or (Any,)
	
	if origin in {list, tuple, set}:
		return (Any,)
	
	if origin == dict:
		return (Any, Any)
	return ()

def generic_origin(t: type):
	'''Return the origin of a type, even if it's not a generic.'''
	
	return typing.get_origin(t) or t

def unpack_generic(t: type):
	'''Unpack a generic type into its origin and parameters.'''
	
	return generic_origin(t), generic_args(t)

def unpack_container(schema: type):
	'''
	Given a manifest type, unpack the relevant schema types to return a union of valid element types.
	
	Example: unpack_container(list, dict[str, float]|list[bool]|list[int|str]) -> bool|int|str
	'''
	
	origin, args = unpack_generic(schema)
	if origin in {Union, types.UnionType}:
		return tuple(e for opt in args for e in typing.get_args(opt))
	
	return args

BRACE = {
	"(": ")",
	"[": "]",
	"{": "}",
}

def parse_value(target, schema):
	'''
	ParseState must contain:
	* state - FDA state label
	* current brace ( [ { quote
	* allowed element type set/list, need to account for Any
	
	* maybe store bits for each brace type whether the element type is sensitive to that brace
	
	Building strings:
	* Quoted strings are always self-contained
	* Unquoted strings can be broken by certain special characters like :, these are context-sensitive and when they have no special meaning they must be concatenated to the string. Unquoted strings still emit str_start and str_end events, so 
	'''
	stack: list[ParseState] = []
	
	while True:
		kind, value = yield
		
		match (stack[-1].kind, *(yield)):
			case ("seq"|"doc", "seq_start", kind):
				stack.append(ParseState('seq', kind, BRACE[kind]))
				target.send(("start", kind))
			
			case ("seq"|"doc", "seq_end", kind):
				if stack.pop().close != kind:
					raise ValueError(f"Unmatched brace {kind!r}")
				
				target.send(("end", kind))
			
			case ("seq"|"doc", "str_start", ""):
				stack.append(ParseState('str', quote, ""))
			
			# Unquoted string starts
			case ("seq"|"doc", "str", value):
				stack.append(ParseState('unq', "", value))
			
			case ("unq", "str", value):
				stack[-1].value += value
			case ("unq", ":", _):
				if is_key:
					target.send(("str", stack.pop().value))
					target.send((":", ":"))
			
			# Unquoted string terminates
			case ("unq", "str_start", quote):
				target.send(("str", stack.pop().value))
				stack.append(ParseState('str', quote, ""))
			
			case ("str", "str", value):
				stack[-1].value += value
			
			# When unquoted terminates, we need a special state to see if it should continue
			case ("str", "str_end", ""):
				stack[-1].kind = 'unq'
			
			case ("str", "str_end", quote):
				target.send(("str", stack.pop().value))
			
			case (kind, value):
				if stack[-1].kind == 'str':
					stack[-1].value += value
				else:
					stack.append(ParseState("str", kind, ""))

def parse_string(lex):
	if lex is not None:
		parts = [lex.value]
	
	while True:
		kind, quote, value = yield
		parts.append(value)
		if kind == 'end':
			break
	return quote, ''.join(parts)

def parse_list():
	items = []
	lex = None
	while True:
		value = yield from parse_value(lex)
		items.append(value)
		lex = yield
		
		if lex == ('end', ']', None):
			break
		elif lex == ('atom', ',', None):
			lex = None
	return items

def parse_object():
	obj = {}
	lex = None
	while True:
		key = yield from parse_string(lex)
		lex = yield
		if lex != ('atom', ':', None):
			raise ValueError("Expected ':'")
		value = yield from parse_value(lex)
		obj[key] = value
		lex = yield
		
		if lex == ('end', '}', None):
			break
		elif lex == ('atom', ',', None):
			lex = None
	return obj

def parse_value(lex):
	if lex is None:
		lex = yield
	
	if lex.kind == 'start':
		if lex.quote == "[":
			value = yield from parse_list()
		elif lex.quote == "{":
			value = yield from parse_object()
		else:
			value = yield from parse_string(lex)
	elif lex.kind == 'atom':
		value = lex.value
	else:
		raise ValueError("Unexpected lexeme {lex!r}")
	
	return value

@utils.coroutine
def parse_value(target):
	"""
	Parses results coming out of the Lexer into ijson events, which are sent to
	`target`. A stack keeps track of the type of object being parsed at the time
	(a value, and object or array -- the last two being values themselves).

	A special EOF result coming from the Lexer indicates that no more content is
	expected. This is used to check for incomplete content and raise the
	appropriate exception, which wouldn't be possible if the Lexer simply closed
	this co-routine (either explicitly via .close(), or implicitly by itself
	finishing and decreasing the only reference to the co-routine) since that
	causes a GeneratorExit exception that cannot be replaced with a custom one.
	"""
	
	stack = [_PARSE_VALUE]
	prev_pos, prev_symbol = None, None
	while True:
		if prev_pos is None:
			pos, symbol = yield
			if (pos, symbol) == EOF:
				if stack:
					raise common.IncompleteJSONError('Incomplete JSON content')
				break
		else:
			pos, symbol = prev_pos, prev_symbol
			prev_pos, prev_symbol = None, None
		try:
			state = stack[-1]
		except IndexError:
			if True:
				state = _PARSE_VALUE
				stack.push(state)
			else:
				raise common.JSONError('Additional data found')
		assert stack

		if state == _PARSE_VALUE:
			# Simple, common cases
			if symbol == 'null':
				target.send(('null', None))
				stack.pop()
			elif symbol == 'true':
				target.send(('boolean', True))
				stack.pop()
			elif symbol == 'false':
				target.send(('boolean', False))
				stack.pop()
			elif symbol[0] == '"':
				target.send(('string', parse_string(symbol)))
				stack.pop()
			# Array start
			elif symbol == '[':
				target.send(('start_array', None))
				pos, symbol = yield
				if (pos, symbol) == EOF:
					raise common.IncompleteJSONError('Incomplete JSON content')
				if symbol == ']':
					target.send(('end_array', None))
					stack.pop()
				else:
					prev_pos, prev_symbol = pos, symbol
					stack.push(_PARSE_ARRAY_ELEMENT_END)
					stack.push(_PARSE_VALUE)
			# Object start
			elif symbol == '{':
				target.send(('start_map', None))
				pos, symbol = yield
				if (pos, symbol) == EOF:
					raise common.IncompleteJSONError('Incomplete JSON content')
				if symbol == '}':
					target.send(('end_map', None))
					stack.pop()
				else:
					prev_pos, prev_symbol = pos, symbol
					stack.push(_PARSE_OBJECT_KEY)
			# A number
			else:
				# JSON numbers can't contain leading zeros
				if ((len(symbol) > 1 and symbol[0] == '0' and symbol[1] not in ('e', 'E', '.')) or
					(len(symbol) > 2 and symbol[0:2] == '-0' and symbol[2] not in ('e', 'E', '.'))):
					raise common.JSONError('Invalid JSON number: %s' % (symbol,))
				# Fractions need a leading digit and must be followed by a digit
				if symbol[0] == '.' or symbol[-1] == '.':
					raise common.JSONError('Invalid JSON number: %s' % (symbol,))
				try:
					number = to_number(symbol)
					if number == inf:
						raise common.JSONError("float overflow: %s" % (symbol,))
				except:
					if 'true'.startswith(symbol) or 'false'.startswith(symbol) or 'null'.startswith(symbol):
						raise common.IncompleteJSONError('Incomplete JSON content')
					raise UnexpectedSymbol(symbol, pos)
				else:
					target.send(('number', number))
					stack.pop()

		elif state == _PARSE_OBJECT_KEY:
			if symbol[0] != '"':
				raise UnexpectedSymbol(symbol, pos)
			target.send(('map_key', parse_string(symbol)))
			pos, symbol = yield
			if (pos, symbol) == EOF:
				raise common.IncompleteJSONError('Incomplete JSON content')
			if symbol != ':':
				raise UnexpectedSymbol(symbol, pos)
			stack[-1] = _PARSE_OBJECT_END
			stack.push(_PARSE_VALUE)

		elif state == _PARSE_OBJECT_END:
			if symbol == ',':
				stack[-1] = _PARSE_OBJECT_KEY
			elif symbol != '}':
				raise UnexpectedSymbol(symbol, pos)
			else:
				target.send(('end_map', None))
				stack.pop()
				stack.pop()

		elif state == _PARSE_ARRAY_ELEMENT_END:
			if symbol == ',':
				stack[-1] = _PARSE_ARRAY_ELEMENT_END
				stack.push(_PARSE_VALUE)
			elif symbol != ']':
				raise UnexpectedSymbol(symbol, pos)
			else:
				target.send(('end_array', None))
				stack.pop()
				stack.pop()
