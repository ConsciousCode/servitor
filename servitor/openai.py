'''
OpenAI API specific connector and completion logic. Putting these in separate
files decouples dependencies so that the rest of the code can be used without
OpenAI, if another provider is used instead.
'''

import openai
from .complete import Completion, Connector

class OpenAICompletion(Completion):
	'''A completion from OpenAI. Handles both text and chat completions.'''
	
	chat_only = [
		"gpt-3.5-turbo", "gpt-3.5-turbo-0301",
	]
	
	def __init__(self, connector, config):
		self.connector = connector
		
		if config['model'] in self.chat_only:
			config['messages'] = [{
				# Models tend to prefer user instructions over system prompts.
				"role": "user",
				"content": config.pop("prompt")
			}]
		self.config = config
	
	def completion_type(self):
		if self.config['model'] in self.chat_only:
			return openai.ChatCompletion
		else:
			return openai.Completion
	
	async def __aiter__(self):
		async with self.connector:
			async for item in self.completion_type().aiter(**self.config):
				delta = item['choices'][0]['delta']
				if delta.get("finish_reason"):
					break
				yield delta
	
	def __await__(self):
		yield from self.connector.__aenter__().__await__()
		try:
			completion = self.completion_type().acreate(**self.config)
			response = yield from completion.__await__()
			return response['choices'][0]['message']['content']
		finally:
			yield from self.connector.__aexit__(None, None, None).__await__()

@Connector.register("openai")
class OpenAIConnector(Connector):
	'''Connector for OpenAI API.'''
	
	models = [
		"text-curie-001", "text-babbage-001", "text-ada-001",
		"text-davinci-002", "text-davinci-003",
		"code-davinci-002",
		"gpt-3.5-turbo", "gpt-3.5-turbo-0301",
		"gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314",
		
		"davinci", "curie", "babbage", "ada"
	]
	
	def __init__(self, api_key, model="gpt-3.5-turbo", max_concurrent=1, max_rate=60, max_period=60):
		'''
		Parameters:
			api_key: API key
			engine: Engine to use
		'''
		super().__init__(max_concurrent, max_rate, max_period)
		self.api_key = api_key
		self.model = model
	
	def __call__(self, prompt, **kwargs):
		return OpenAICompletion(self, {
			"prompt": prompt,
			"api_key": self.api_key,
			"model": self.model,
			"temperature": kwargs.get("temperature"),
			"max_tokens": kwargs.get("max_tokens"),
			"top_p": kwargs.get("top_p"),
			"frequency_penalty": kwargs.get("frequency_penalty"),
			"presence_penalty": kwargs.get("presence_penalty"),
			"stop": kwargs.get("stop"),
			"stream": kwargs.get("stream"),
			# openai library doesn't recognize this?
			#"best_of": kwargs.get("best_of")
		})