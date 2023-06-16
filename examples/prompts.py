from servitor import semantic

@semantic
def summarize(text) -> str:
    """Summarize the given text in 2 sentences or less."""

@semantic
def list_people(text) -> list[str]:
    """List the people mentioned in the given text."""

@semantic
def rank_importance(idea: str) -> float:
    """Rank the importance of the given idea on a scale from 0 to 1."""

@semantic
def to_command(query: str) -> str:
    """Convert the given query into a bash command."""

@semantic
def language(text: str) -> str:
	"""Detect the human language of the given text. The return should be all lowercase."""

@semantic
def translate(context: str, text: str, language: str) -> str:
	"""Translate the given text into the given language. Additional context can be provided to disambiguate the translation."""

@semantic
def pluralize(word) -> str:
	"""Pluralize the given word."""

@semantic
def correct_i18n(text) -> str:
	"""The given text used a basic substitution i18n. Correct any grammatical mistakes."""

@semantic
def normalize_json(error: str, json: str) -> object:
	"""The given JSON failed to parse. Attempt to correct all parsing errors."""

@semantic
def critique_code(code: str) -> str:
	"""Find as many issues in the given code as you can and return a comprehensive critique along with suggestions for improvement."""

@semantic
def improve_code(criticism: str, code: str) -> str:
	"""Improve the given code based on the given criticism."""

@semantic
def correct_code(error: list[str], code: str) -> str:
	"""The code failed to compile due to the given errors. Attempt to correct all parsing errors, even if they don't appear in the list."""

@semantic
def generate_code(task: str) -> str:
	"""Generate Python code that will complete the given task."""

@semantic
def diff_comment(diffs: list[str]) -> str:
	"""Given a list of diffs in a git commit, generate a comment that summarizes the changes."""

@semantic
def audit_yaml(yaml: str) -> list[str]:
	"""Given a YAML file, audit it for security vulnerabilities and return a list of issues."""

@semantic
def mud(user: dict, env: dict, command: str) -> tuple[str, dict]:
	"""Given a user, environment, and command, return the output of the command and updates to the new environment."""