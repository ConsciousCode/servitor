[Main](../../)

This project is licensed under the [MIT license](../../LICENSE).

Servitor is a library for creating "semantic functions".

## Files
* [](__init__.py) - Public-facing interface, including `semantic` decorator.
* [](kernel.py) - `Kernel` base class and `DefaultKernel` 
* [](adapter.py) - `Adapter` base class and default implementations.
* [](complete.py) - `Connector`, `Completion`, `Delta`, and logic for lazily loading provider connectors.
* [](connector/) - Connectors for various providers.
* [](util.py) - Various utilities and error classes.
