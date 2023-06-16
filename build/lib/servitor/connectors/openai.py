'''
OpenAI REST API connector and completion.
'''

import openai
import tiktoken
from . import Completion, Connector, Throttle
from ..util import logger

logger.info("import", extra={"module": "openai"})

# Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def chat_token_count(enc, msgs, model):
	"""Returns the number of tokens used by a list of messages."""
	try:
		enc = tiktoken.encoding_for_model(model)
	except KeyError:
		enc = tiktoken.get_encoding("cl100k_base")
	
	

class OpenAICompletion(Completion):
	'''A completion from OpenAI. Handles both text and chat completions.'''
	
	# Maps chat-only models to (per_msg, per_name) token counts
	chat_only = {
		"gpt-3.5-turbo": (4, -1),
		"gpt-3.5-turbo-0301": (4, -1),
		"gpt-4": (3, 1),
		"gpt-4-0314": (3, 1),
		"gpt-4-0613": (3, 1),
	}
	
	def __init__(self, connector, config):
		self.connector = connector
		self.config = config
		
		try:
			enc = tiktoken.encoding_for_model(config['model'])
		except KeyError:
			enc = tiktoken.encoding_for_model("cl100k_base")
		
		model = config['model']
		
		if model in self.chat_only:
			msgs = [{
				# Models tend to prefer user instructions over system prompts.
				"role": "user",
				"content": config.pop("prompt")
			}]
			config['messages'] = msgs
			
			per_msg, per_name = self.chat_only[model]
			
			tokens = len(msgs)*per_msg + 3
			for msg in msgs:
				for key, value in msg.items():
					tokens += len(enc.encode(value))
					if key == "name":
						tokens += per_name
		else:
			tokens = len(enc.encode(config['prompt']))
		
		self.token_count = tokens
	
	def completion_type(self):
		if self.config['model'] in self.chat_only:
			return openai.ChatCompletion
		else:
			return openai.Completion
	
	async def __aiter__(self):
		async with self.connector.throttle(self.token_count):
			async for item in self.completion_type().aiter(**self.config):
				delta = item['choices'][0]['delta']
				if delta.get("finish_reason"):
					break
				self.connector.throttle.add_tokens(1)
				yield delta
	
	def __await__(self):
		async def _body(self):
			async with self.connector.throttle(self.token_count):
				res = await self.completion_type().acreate(**self.config)
				self.connector.throttle.add_tokens(res['usage']['completion_tokens'])
				return res['choices'][0]['message']['content']
		yield from _body(self).__await__()

@Connector.register("openai")
class OpenAIConnector(Connector):
	'''Connector for OpenAI API.'''
	
	models = None
	
	def __init__(self, *, openai_api_key=None, model=None, max_concurrent=1, max_rate=60, max_period=60, **other):
		'''
		Parameters:
			api_key: API key
			engine: Engine to use
		'''
		super().__init__()
		self.api_key = openai_api_key
		self.model = model
		self.throttle = Throttle(max_rate, max_period, max_concurrent)
	
	def __call__(self, prompt, **kwargs):
		model = kwargs.get("model", self.model)
		if model is None:
			raise ValueError("No model specified")
		
		return OpenAICompletion(self, {
			"prompt": prompt,
			"api_key": self.api_key,
			"model": model,
			"temperature": kwargs.get("temperature"),
			"max_tokens": kwargs.get("max_tokens"),
			"top_p": kwargs.get("top_p"),
			"frequency_penalty": kwargs.get("frequency_penalty"),
			"presence_penalty": kwargs.get("presence_penalty"),
			"stop": kwargs.get("stop"),
			# openai library doesn't recognize this?
			#"best_of": kwargs.get("best_of")
		})
	
	@classmethod
	def supports(cls, key):
		if cls.models is None:
			cls.models = [model['id'] for model in openai.Engine.list()['data']]
		return key in cls.models