Properties
##########

.. module:: mlacs.properties

.. index::
   single: Class reference; PropertyManager

PropertyManager
***************

The PropertyManager Class in an object to manage the calculation on the fly of particular properties. 
The training of the MLIP stops when all properties are converged according to user defined criterions.
The Property manager takes as input a list of :class:`CalcProperty`.
During a MLACS step, the Property manager start to initalize the :class:`CalcProperty` objects, then it runs the different objects and finaly it checks if all the objects achieve their convergence criterion. In this case, the MLACS simulation can be stopped.

.. autoclass:: PropertyManager
   :members: calc_initialize, run, check_criterion

CalcProperty
************

The CalcProperty Classes are properties you want to compute at some point durinf the MLACS simulation.
The :class:`CalcProperty` objects can different object as input (states, atoms, functions, ...).

.. autoclass:: CalcProperty
   :members: isconverged

CalcPafi
~~~~~~~~

.. autoclass:: CalcPafi

CalcNeb
~~~~~~~

.. autoclass:: CalcNeb

CalcTi
~~~~~~

.. autoclass:: CalcTi

CalcRdf
~~~~~~~

.. autoclass:: CalcRdf

CalcAdf
~~~~~~~

.. autoclass:: CalcAdf

CalcExecFunction
~~~~~~~~~~~~~~~~

.. autoclass:: CalcExecFunction
