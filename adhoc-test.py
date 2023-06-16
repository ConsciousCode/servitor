'''
Ad-hoc testing - random code to test things out without having to write whole
unit tests.
'''

from servitor import semantic

import sys
import logging
logging.basicConfig(
	level=logging.DEBUG,
	stream=sys.stdout,
	format='%(asctime)s %(message)s'
)

@semantic
def list_people(text) -> list[str]:
	"""List people mentioned in the given text."""

print(list_people("John and Mary went to the store."))

@semantic
def mud_charbuild(name: str) -> dict:
	"""Given a name, return a character dictionary for a MUD."""
print(mud_charbuild("John"))