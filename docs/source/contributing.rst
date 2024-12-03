.. _contributing:
.. index:: Contributing

Contributing
############

Github repository
=================

All work on the source code takes place on Github_.

.. _Github: https://github.com/mlacs-developers/mlacs

The first steps on github
-------------------------

* Register as a user on https://about.github.com/
* Find and/or change your user-name in your account setting. You will need it.
* Follow directions on https://github.com/help/ssh/README for how to generate
  and add your own public ssh-key
* Go to https://github.com/mlacs-developers/mlacs and fork the project. 
  The 'fork' button is between the watch and the star button. Click on
  it then deselect the option 'Copy the main branch only' to have access to
  the development branches.
  Forking the project give you your own personal copy of the project.

You will now have a fork situated at https://github.com/your-user-name/mlacs/

* From here on:

  - ``upstream`` refers to git@gitlab.com:ase/ase and refers to the official
    repository  of the ase project.
  - ``origin`` refers to your copy of the ase project located at
    git@github.com:your-user-name/mlacs or
    https://github.com/your-user-name/mlacs/

  For example, ``upstream/main`` refers to the main (i.e., trunk or
  development) branch of the official ase project.

* Clone your fork to ``origin`` to your local machine

.. code:: bash

      $ git clone git@gitlab.com:your-user-name/mlacs.git
      $ cd ase

* Add and track ``upstream`` in your local git (done only once for each local
  git repository)

.. code:: bash

      $ git remote add upstream git@github.com:mlacs-developers/mlacs.git

You can always check the list of remote repositories that you can obtain
code from

.. code:: bash

      $ git remote -v

And you can check all available branches from the remotes that you are
tracking

.. code:: bash

      $ git branch -a

Merge your developments
-----------------------

* First you need to synchronise your fork.
* Resolve possible conflicts.
* Then click on 'Contribute' then 'Open pull request'.
* Provide informative title and more verbose description in the body of the
  Pull Request. Do not hold back in the description of the pull request and be
  as explicit as possible.
* Make sure the Pull Request is targeting a development branch (at least the
  develop branch).
* Wait for feedback from the developer community and address concerns as
  needed by adding more commits to the your branch on your
  personal repository and then pushing to your github repository.
* Once the developer community is satisfied with your merge request,
  anyone with push access can merge your pull request and it will now be part
  of the main branch

* After the merge-request is approved, delete the branch locally

.. code:: bash

        $ git branch -D add-contribute-rst

Code review
===========

Before you start working on a Merge Request, *please* read our coding rules. 

Mlacs provides different levels of tests. To run the tests, you should install
pytest first.

.. code:: bash

      $ python -m pip install pytest

Then you can run the tests using the commands.

.. code:: bash

      $ pytest --fast         # For a quick review of your development
      $ pytest                # Will run all the tests
      $ pytest --full         # Will run all the tests and examples

All the tests and examples (option full) should be working before considering
to merge.

Hopefully someone will look at your changes and give you some
feedback.  Maybe everything is fine and things can be merged to the official
repository right away, but there could also be some more work to do like:

* make it compatible with all supported Pythons (see
  :ref:`installation`).
* write more comments
* fix docstrings
* write a test
* add some documentation

Such code review is practiced by virtually all software projects
that involve more than one person. Code review should be viewed as an
opportunity for you to learn how to write code that fits into the ASE codebase.

Coding Rules
============

The code must be compatible with the oldest supported version of python
as given on the :ref:`installation` page.

Please run Flake on your code.
    
.. code:: bash

    $ flake8 your_code.py


The rules are almost identical
to those used by the Docutils project:

Contributed code will not be refused merely because it does not
strictly adhere to these conditions; as long as it's internally
consistent, clean, and correct, it probably will be accepted.  But
don't be surprised if the "offending" code gets fiddled over time to
conform to these conventions.

The project follows the generic coding conventions as
specified in the Style Guide for Python Code and Docstring
Conventions PEPs, clarified and extended as follows:

* Do not use "``*``" imports such as ``from module import *``.  Instead,
  list imports explicitly.

* Use 4 spaces per indentation level.  No tabs.

* Read the *Whitespace in Expressions and Statements*
  section of PEP8.

* Avoid trailing whitespaces.

* No one-liner compound statements (i.e., no ``if x: return``: use two
  lines).

* Maximum line length is 78 characters.

* Use "StudlyCaps" for class names.

* Use "lowercase" or "lowercase_with_underscores" for function,
  method, and variable names.  For short names,
  joined lowercase may be used (e.g. "tagname").  Choose what is most
  readable.

* No single-character variable names, except indices in loops
  that encompass a very small number of lines
  (``for i in range(5): ...``).

* Avoid lambda expressions.  Use named functions instead.

* Avoid functional constructs (filter, map, etc.).  Use list
  comprehensions instead.

* Use ``'single quotes'`` for string literals, and ``"""triple double
  quotes"""`` for docstrings.  Double quotes are OK for
  something like ``"don't"``.
