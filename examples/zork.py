'''
Implements a bare-bones zork using semantic functions. The LLM does not see the
conversation history! It only sees the current state of the world and the
current command. It is up to the semantic functions to maintain state.
'''

import sys
import os
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))
'''
import logging

logger = logging.getLogger("servitor")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s')
logger.addHandler(handler)
#'''
from typing import Mapping
import json
from dataclasses import dataclass, asdict

from servitor import semantic

def merge(d, u):
	for k, v in u.items():
		if (k in d and isinstance(d[k], Mapping) and isinstance(v, Mapping)):
			merge(d[k], v)
		elif v is None:
			if k in d:
				del d[k]
		else:
			d[k] = v

@semantic(temperature=1.2, top_p=0.2)
def zork_world() -> str:
	"""Return a description of a world and initial location for a text adventure."""
	print("Building world...")

@semantic
def zork_charbuild(name: str) -> dict:
	"""Given a name, return a character dictionary for a text adventure."""
	print("Building character...")

@semantic
def zork_cmd(env: dict, history: list[str], cmd: str) -> tuple[str, dict]:
	"""Given an environment and command, return the command response."""

@semantic
def zork_update(env: dict, cmd: str, response: str) -> dict:
	"""Given an environment, command, and its response, return updates to the environment. Set values to null to delete them. This is stateless, so it adds important details to remember in future commands."""

HISTORY_LEN = 10
SAVE_FILE = "private/zork.json"

@dataclass
class State:
	env: dict
	history: list[tuple[str, str]]
	
	def add_cmd(self, cmd, response):
		self.history.append((cmd, response))
		self.history = self.history[-HISTORY_LEN:]

def main():
	try:
		with open(SAVE_FILE, "r") as f:
			state = State(**json.load(f))
	except (json.JSONDecodeError, ValueError, TypeError, FileNotFoundError):
		state = None
	
	if state is None:
		name = input("What is your name, traveler? ")
		prompt = zork_world()
		print("World:", prompt)
		user = zork_charbuild(name)
		
		state = State({"user": user, "prompt": prompt}, [])
	
	while True:
		cmd = input("> ")
		if cmd.startswith("/"):
			cmd, *args = cmd[1:].split(" ")
			match cmd:
				case "h"|"help":
					print("Commands:\n  /h, /help: Show this help message.\n  /q, /quit, /exit: Quit the game.\n  /env: Print the environment.")
				
				case "q"|"quit"|"exit":
					return
				
				case 'env':
					print(f"{state=}")
				
				case _:
					print(f"Unknown command: {cmd}")
		else:
			out = zork_cmd(state.env, state.history, cmd)
			print(out)
			state.add_cmd(cmd, out)
			merge(state.env, zork_update(state.env, cmd, out))
			
			with open("zork.json", "w") as f:
				json.dump(asdict(state), f)

if __name__ == "__main__":
	main()