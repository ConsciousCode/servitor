import openai
import unittest
import asyncio
import sys
sys.path.append("src")
from servitor import semantic, SemanticFunction, Kernel, ChainOfThought, SemanticFunction
import servitor

class TestServitor(unittest.TestCase):

    def test_semantic_function(self):
        return
        @semantic
        def list_people(text) -> list[str]:
            """List people mentioned in the given text."""

        self.assertIsInstance(list_people, SemanticFunction)

        people = list_people("John and Mary went to the store.")
        
        self.assertEqual(people, ["John", "Mary"])

    def test_custom_prompt(self):
        return
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
        @semantic(adapter="chain")
        async def summarize_async(concept: str) -> str:
            """Summarize the given text in two sentences or less."""
        
        summary = asyncio.run(summarize_async("Long text about a subject that can be summarized in two sentences."))
        self.assertIsInstance(summary, ChainOfThought)
        self.assertTrue(len(summary.answer.split(". ")) <= 2)
    
    def test_raw_openai(self):
        return
        async def test():
            from servitor.connectors.openai import OpenAIConnector
            api_key = "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"
            oaic = OpenAIConnector()
            return await oaic.complete("1,2,3,", servitor.config.DefaultConfig(dict(
                openai_api_key=api_key,
                model='davinci-002',
                prompt='1,2,3,',
                max_tokens=1,
                temperature=0
            ), servitor.defaults.config))
        
        print("Raw", asyncio.run(test()))

if __name__ == '__main__':
    unittest.main()
