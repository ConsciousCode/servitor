'''
Default configurations and values.
'''

openai = dict(
	model = "gpt-3.5-turbo"
)

config = dict(
	# OpenAI
	openai_api_key = None,
	openai_organization = None,

	# GPT4All
	model_path = None,
	allow_download = True,

	# Common parameters
	model = None,

	# Generation parameters
	adapter = "chain",
	temperature = 0,
	top_p = 0.9,
    top_k = 0,
	frequency_penalty = 0,
	presence_penalty = 0,
	max_tokens = 1000,
	best_of = 1,

	# Throttling
	retry = 3,
	concurrent = 1,
	request_rate = 60,
	token_rate = 250000,
	period = 60
)