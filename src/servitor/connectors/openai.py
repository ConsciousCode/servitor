'''
OpenAI REST API connector and completion.
'''

import openai
import openai.error
import tiktoken
from contextlib import contextmanager
from functools import cached_property

from . import Throttle, Connector
from .. import defaults
from ..util import logger, async_await, BusyError, ThrottleError
from ..typings import override

logger.info("Import OpenAI connector")

@contextmanager
def transmute_errors():
	'''Convert errors we may need to catch to our own exceptions.'''
	try:
		yield
	except openai.error.RateLimitError as e:
		raise ThrottleError() from e
	except (openai.error.ServiceUnavailableError, openai.error.TryAgain) as e:
		raise BusyError() from e

class OpenAICompletion:
	'''A completion from OpenAI. Handles both text and chat completions.'''
	
	def __init__(self, prompt, throttle, config):
		self.prompt = prompt
		self.throttle = throttle
		self.config = config
		
		try:
			enc = tiktoken.encoding_for_model(config['model'])
		except KeyError:
			enc = tiktoken.encoding_for_model("cl100k_base")
		
		model = config['model']
		
		gpt3_5 = model.startswith("gpt-3.5")
		gpt4 = model.startswith("gpt-4")
		
		# Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
		if gpt3_5 or gpt4:
			if gpt3_5:
				per_msg, per_name = 4, -1
			else:
				per_msg, per_name = 3, 1
				
			msgs = [{
				# Models tend to prefer user instructions over system prompts.
				"role": "user",
				"content": prompt
			}]
			config['messages'] = msgs
			del config['prompt']
			
			tokens = len(msgs)*per_msg + 3
			for msg in msgs:
				for key, value in msg.items():
					tokens += len(enc.encode(value))
					if key == "name":
						tokens += per_name
		else:
			tokens = len(enc.encode(prompt))
		
		self.tokens = tokens
	
	def is_chat_model(self):
		'''Return whether or not the model is a chat model.'''
		model = self.config['model']
		return model.startswith("gpt-3.5") or model.startswith("gpt-4")

	def completion_type(self):
		'''Get the openai endpoint for creating a completion.'''
		return openai.ChatCompletion if self.is_chat_model() else openai.Completion
	
	def build_async_completion(self, stream):
		'''Call acreate() on the right completion type.'''
		return self.completion_type().acreate(
			stream=stream,
			**self.config
		)
	
	def unpack_content(self, result):
		'''Unpack the content of a result.'''

		self.throttle.output(result['usage']['completion_tokens'])
		result = result['choices'][0]
		if self.is_chat_model():
			return result['message']['content']
		else:
			return result['text']
	
	@override
	async def __iter__(self):
		with transmute_errors():
			with self.throttle.lock(self.tokens):
				for item in self.completion_type().create(stream=True, **self.config):
					delta = item['choices'][0]['delta']
					if delta.get("finish_reason"):
						break
					self.throttle.output(1)
					yield delta
	
	@override
	async def __aiter__(self):
		with transmute_errors():
			async with self.throttle.alock(self.tokens):
				async for item in await self.build_async_completion(True):
					delta = item['choices'][0]['delta']
					if delta.get("finish_reason"):
						break
					self.throttle.output(1)
					yield delta
	
	@override
	def __call__(self):
		with transmute_errors():
			with self.throttle.lock(self.tokens):
				return self.unpack_content(self.completion_type().create(**self.config))
	
	@override
	@async_await
	async def __await__(self):
		with transmute_errors():
			async with self.throttle.alock(self.tokens):
				return self.unpack_content(await self.build_async_completion(False))

@Connector.register("openai")
class OpenAIConnector(Connector):
	throttle: dict[int, Throttle]

	def __init__(self):
		super().__init__()
		
		self.throttle = {}
	
	@cached_property
	def model_list(self):
		return [model['id'] for model in openai.Engine.list()['data']]
	
	@override
	def supports(self, config):
		return config.get("openai_api_key") or config.get("model") in self.model_list
	
	@override
	def complete(self, prompt, config):
		# Use hash to reduce potential for leaking API keys
		api_key = hash(config['openai_api_key'])
		if api_key not in self.throttle:
			self.throttle[api_key] = Throttle(
				config["request_rate"],
				config["token_rate"],
				config["period"],
				config["concurrent"]
			)
		
		if config['top_k']:
			logger.warn("OpenAI does not support top_k, ignoring")
		
		# TODO: api_base, api_type_, request_id, api_version
		# TODO: best_of (my version of OpenAI doesn't support it)
		return OpenAICompletion(
			prompt,
			self.throttle[api_key], {
			# OpenAI is sensitive to unknown parameters, so we only pass the ones we need.
			"api_key": config.get("openai_api_key"),
			"model": config.get("model", defaults.openai['model']),
			"prompt": prompt,
			"organization": config.get("openai_organization"),
			"temperature": config["temperature"],
			"top_p": config["top_p"],
			"frequency_penalty": config["frequency_penalty"],
			"presence_penalty": config["presence_penalty"],
			"max_tokens": config["max_tokens"],
			"stop": config.get("stop")
		})