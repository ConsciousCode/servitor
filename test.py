from servitor import semantic
import asyncio

longstr = """First, we must define AGI. One prevailing definition of AGI is the ability of an intelligent agent to understand or learn any intellectual task that a human being can. There are many permutations of this definition, but the consensus seems to be human-level learning and comprehension.

We have plenty of expectations about AGI because of science fiction. Intelligent automatons have fascinated humans for thousands of years, from the Minoan legend of Talos to HAL 9000 and Command Data. But these are all works of fiction, so what will AGI really look like when it arrives?

Let us start by breaking down the components of intelligence – what are the internal and external behaviors that we expect of intelligent entities? I found that Bloom’s taxonomy, a system developed for teaching, is a good way to break down the behaviors and outcomes of learning. After all, the current definition of AGI is the ability to learn and understand anything a human can. In descending order of complexity and difficulty, Bloom’s taxonomy is:

Create – Produce new or original work.
Evaluate – Justify a stand or decision.
Analyze – Draw connections among ideas.
Apply – Use information in new situations.
Understand – Explain ideas or concepts.
Remember – Recall facts and basic concepts.
Ultimately, then, AGI must be able to remember facts and concepts, and eventually work its way up to the ability to create new and original work. Under what conditions do we expect the AGI to achieve these goals? Humans generally require study, instruction, and practice – so it may be natural to assume, as Alan Turing did, that an AGI would need to go through the same pedagogical process of learning that humans do; humans learn spontaneously from the time we first open our eyes. Learning and curiosity are intrinsic to our organism, but then we also have institutions of structured learning as well as the ability to self-educate with a variety of tools, such as books and online courses. In any case, humans cannot help but learn from exposure and practice. Our brains are hardwired to ingest new information and integrate it into our models of the world, with or without any conscious effort.

Thus, spontaneous learning is one of the key features we should expect of AGI – whatever else the AGI does, it must learn and adapt all on its own, completely automatically. Even in fictional examples, whether you are talking about Skynet or Commander Data, we all seem to agree that a truly intelligent machine must learn from its experiences in real-time. The AGI might not know everything from the moment you turn it on, but it should be capable of learning anything over time.

Spontaneous learning is a strictly internal, cognitive behavior. Learning needs to happen for something to be intelligent but how would an AGI go about demonstrating its intelligence? We humans often hold creation up as the pinnacle of intelligence and achievement, hence creation’s place at the top of Bloom’s taxonomy. Experts and geniuses are expected to write a book, compose a symphony, or propose a theory of cosmology. Thus, an AGI must be able to produce something, a creation of some sort, to demonstrate its intelligence.

These two behaviors, spontaneous learning and creation, are the goalposts I have set for myself. My architecture, NLCA, must be able to spontaneously learn anything and, ultimately, be able to create novel and valuable intellectual output. Right now, the most sophisticated AI systems can win at any game, yet we have no AI capable of building games. An AGI should be able to learn to play any game, but also deconstruct the process of writing a game, write the code for the game, and test the game."""

@semantic
def summarize(text) -> str:
	'''Summarize text in two sentences.'''

print(asyncio.run(summarize(longstr)))