# jax_chmc
An implementation of a family of MCMC methods on implicitly defined manifolds


The api is similar to the one from [blackjax](https://github.com/blackjax-devs/blackjax) but is slightly simplified, and closer to the original MATLAB [implementation](https://www.cs.toronto.edu/~mbrubake/projects/cmcmc/) by @mbrubake .


#### Project Status

__jax_chmc__ is still in its early phase. We are actively improving various software components. 
It is still quite far from ready for everyday use and is made
available without any support at the moment.


## Installation

You can install jax_chmc via `pip`:

```bash
$ pip install jax_chmc
```



## Development

To develop and modify jax_chmc, you need to install
[`hatch`]([https://python-poetry.org/](https://hatch.pypa.io)), a tool for Python packaging and
dependency management.

To  enter a virtual environment for testing or debugging, you can run:

```bash
$ hatch shell
```

### Running tests

Gsd uses [Pytest](https://pytest.org/) for testing. To run the tests, use the following command:

```
$ hatch run test 
```