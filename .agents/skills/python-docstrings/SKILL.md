---
name: python-docstrings
description: Enforces Google-style Python docstrings for Python code
---

# Python Docstring Rules

Use this skill when reviewing or writing Python docstrings.

## Purpose and scope

- Use Google-style docstrings.
- Focus on public Python modules, classes, functions, and methods.
- Keep docstrings compact, specific, and easy to scan.

## Request changes when

- a public API has no docstring;
- the summary line is vague or inaccurate;
- arguments, returns, or intentionally raised exceptions are undocumented;
- a non-trivial public API needs an example but does not have one.

## Required structure

1. Short description
2. Optional longer explanation
3. `Args`
4. `Returns`
5. Optional `Raises`
6. Optional `Example`

## Formatting rules

- Limit docstrings to 120 characters per line.
- In `Args`, use `name (type): description`.
- In `Returns`, describe both the type and meaning of the returned value.
- Add `Raises` when the function intentionally raises exceptions.
- Use doctest-style `Example` blocks with `>>>` when examples help clarify usage.

## Reviewer checklist

- Is the docstring present on the public API?
- Is the summary line accurate?
- Are inputs, outputs, and exceptions documented?
- Would a user understand how to call this API from the docstring alone?

## Example

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """Short description.

    A longer explanation.

    Args:
        param1 (int): Explanation of param1.
        param2 (str): Explanation of param2. Defaults to "default".

    Returns:
        bool: Explanation of the return value.

    Example:
        >>> my_function(1, "test")
        True
    """
    return True
```

## Docstring for `__init__` method

- Document constructor arguments in the class docstring rather than in a separate `__init__` docstring.

```python
class MyClass:
    """My class description.

    A longer explanation.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.
    Example:
        >>> my_class = MyClass(param1=1, param2="test")
        >>> my_class.param1
        1
        >>> my_class.param2
        'test'
    """
    def __init__(self, param1: int, param2: str) -> None:
        ...
```
