'''
Round robin task scheduling, only need to schedule agents if they have a
message pushed to them. This includes self-messages, useful for maintaining
a self-dialogue and deliberate on something. Multiple pushes can happen,
a single pull will cover all pushes. Once a pull is started, this becomes
a future periodically polled until it's finished, then it's acted upon. This
allows pulls to be atomic, and pushes don't require synchronization.
'''

import asyncio
from collections import defaultdict

class MessageSchema:
    '''
    Schema for parsing and serializing messages for interoperation with
    text-based language models.
    '''

    def parse(self, msg):
        pass

    def serialize(self, msg):
        pass

class Message(Protocol):
    '''
    A single message within a conversation.
    '''

    role: str
    name: str
    content: str

class ChatLog(Protocol):
    '''Abstract interface for chat logs.'''

    def push(self, msg: Message):
        '''Push a message to the chatlog.'''

    def format(self) -> str:
        '''Serialize the chatlog as something readable to the LLM.'''
    
    def consume(self) -> int:
        '''
        Attempt to consume any pending messages.

        Return: The number of messages, or 0 if none are new.
        '''

class Agent:
    '''
    An abstract interface for agents within an AI.
    '''

    name: str
    '''Name of the agent (for debug - not used in messages)'''
    log: ChatLog
    '''The persistent chat log of the agent.'''

    def __init__(self, name: str, log: ChatLog):
        self.name = name
        self.log = log

    def push(self, msg: Message):
        self.log.push(msg)

    async def run(self):
        '''Run the agent in a loop, waiting until at least one new message is pushed.'''

        while True:
            if self.log.consume() == 0:
                await asyncio.sleep(0)
                continue

            self.kernel.


class Logger:
    def push(self, src, dst, msg):
        pass

    def northbus(self, index, msg):
        pass
    
    def southbus(self, index, msg):
        pass

class ACEAgent:
    '''
    An agent which implements a more general structure than the typical ACE,
    providing any number of subagent layers and a north and south bound bus.
    '''

    def __init__(self, agents):
        self._agents = agents
        self._subscribers = defaultdict(list)
    
    def subscribe(self, index, subscriber):
        self._subscribers[index].append(subscriber)
    
    def push_layer(self, index, msg):
        '''Push to an agent at the given layer.'''
        self._agents[index].push(msg)
    
    def log_push(self, src, dst, msg):
        '''Log a push from src to dst.'''
        raise NotImplementedError()
    
    def log_northbus(self, index, msg):
        '''Log a push to the northbus.'''
        self.log_push(index, msg)
    
    def log_southbus(self, index, msg):
        '''Log a push to the southbus.'''
        self.log_push(index, msg)
    
    def northbus(self, index, msg):
        '''Push to every layer at or above the given index.'''
        self.log_northbus(index, msg)
        for i in range(len(layers) - index, len(layers)):
            self.push_layer(-i, msg)
    
    def southbus(self, index, msg):
        '''Push to every layer at or below the given index.'''
        self.log_southbus(index, msg)
        for i in range(max(len(layers), index)):
            self.push_layer(i, msg)

    async def _layer(self, index, agent):
        '''
        To make ACE work, each layer must keep track of its queues which is out
        of scope for ordinary agents.
        '''

        while True:
            completion = await agent.pull(msg)
            msg = self.schema.deserialize(completion)
            match msg.dst:
                case "north": self.northbus(msg)
                case "south": self.southbus(msg)
                case "self": self.push_layer(index, None)
                case "up": self.push_layer(index - 1, msg)
                case "down": self.push_layer(index + 1, msg)

                case _:
                    agent = self._agents[index]
                    await asyncio.gather(ob(agent, msg) for ob in self._subscribers[index])
    
    def start(self):
        '''Build the tasks to run the ACE agent and return the task group.'''
        
        tg = asyncio.TaskGroup()
        for index, agent in enumerate(self._agents):
            tg.create_task(self._layer(index, agent))
        
        return tg