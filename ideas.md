
			
I need to write a custom parser for a JSON-based, YAML-like format optimized to normalize common mistakes made by LLMs when prompted to generate structured data. Some of the common typos I've seen are:
* Missing commas, especially at the end of lines
* Missing quotes around strings
* Variable keywords (eg null / none / None)
* Emitting parenthesized tuples when asked for a tuple in a JSON context, instead of lists

Some of these errors result in ambiguities, so the parser's output will need to represent the ambiguous types. The output of the format is generally typed by the semantic function's signature given to the LLM, so these ambiguities can be resolved by the context provided by the semantic function. For example, if the semantic function expects a list of strings, then the parser can assume that a parenthesized tuple is a list of strings.

In addition to these common mistakes, LLMs are slow enough that we need to stream their outputs token by token to appear responsive to a user - in this case, I want to be able to stream a subfield of the structured data, or stop the LLM early if it's producing malformed output, which virtually no other parser supports. 

Agent
    Connector
    chatlog
    prompt
    context

class Agent:
    def __init__(self):
        self.prompt = ""
        self.context = ""
        self.chatlog = []
    
    def push(self, message):
        '''Push a message to the agent.'''
        pass
    
    def pull(self):
        '''Request a response from the agent.'''