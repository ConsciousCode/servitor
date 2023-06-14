[Main](../)

This project is licensed under the MIT license. See [LICENSE](../LICENSE) for more details.

Servitor is a library for creating "semantic functions".

## Files
* **__init__.py** - Public-facing interface.
* **adapter.py** - `Adapter` base class and default implementations.
* **complete.py** - `Connector`, `Completion`, `Delta`, and logic for lazily loading provider connectors.
* **kernel.py** - `Kernel` base class, `DefaultKernel` logic, `semantic` instance, and `optargs_decorator`.
* **openai.py** - `OpenAIConnector` and `OpenAICompletion` classes, only file that imports `openai`.
* **util.py** - `default`, errors, `Registry`, and `build_task`.
* **test-template.txt** - `TypeAdapter` prompt template for testing in a playground interface.
