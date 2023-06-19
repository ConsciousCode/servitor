import unittest
from asyncio import run
import sys
sys.path.append("src")
from servitor import semantic, Kernel, ChainOfThought

class TestServitor(unittest.TestCase):

    def test_semantic_function(self):
        @semantic
        def list_people(text) -> list[str]:
            """List people mentioned in the given text."""

        people = list_people("John and Mary went to the store.")
        self.assertEqual(people, ["John", "Mary"])

    def test_custom_prompt(self):
        @semantic
        def classify_valence(text: str) -> float:
            return """Classify the valence of the given text as a value between -1 and 1."""

        valence = classify_valence("I am happy.")
        self.assertTrue(0 <= valence <= 1)

    def test_plain_adapter(self):
        @semantic(adapter="plain")
        def summarize(text) -> str:
            """Summarize the given text in two sentences or less."""

        summary = summarize("Long text about a subject that can be summarized in two sentences.")
        self.assertTrue(len(summary.split(". ")) <= 2)

    def test_chain_of_thought_adapter(self):
        @semantic(adapter="cot")
        async def summarize_async(concept: str) -> str:
            """Summarize the given text in two sentences or less."""

        summary = run(summarize_async("Long text about a subject that can be summarized in two sentences."))
        self.assertIsInstance(summary, ChainOfThought)
        self.assertTrue(len(summary.answer.split(". ")) <= 2)

if __name__ == '__main__':
    unittest.main()
