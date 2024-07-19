.. _background:
.. index:: Background


Background
##########


Considering an arbitrary system of :math:`N_\mathrm{at}` atoms at a temperature :math:`T`, and subject to a potential :math:`V(\mathbf{R})`, the classical average of an observable :math:`O(\mathbf{R})` is written

.. math::
   \langle O(\mathbf{R}) \rangle = \int d \mathbf{R}  O(\mathbf{R}) p(\mathbf{R}) \approx \sum_n w_n O(\mathbf{R}_n)

where :math:`p(\mathbf{R}) = e^{-\beta V(\mathbf{R})}/\mathcal{Z}` is the Boltzmann weight, with :math:`\mathcal{Z}=\int e^{-\beta V(\mathbf{R})}` the partition function and :math:`w_i` is the weight of configuration :math:`i` in the approximation involving a limited number of sample.

In the context of *ab initio* simulations, obtaining the canonical distribution :math:`p(\mathbf{R})` entails costly *Ab Initio* Molecular Dynamics (AIMD), which can be challenging to perform or even beyond reach.

The goal of the Machine-Learning Assisted Canonical Sampling approach is to produce a reduced set of configurations and their associated weight using a Machine-Learning Interatomic Potential (MLIP) in order to approximate the canonical distribution and allow the computation of finite-temperature property in an *ab initio* setting.


Theory
******

Machine-Learning Assisted Canonical Sampling
--------------------------------------------

MLACS is built upon the Kullback-Leibler divergence, whose goal is to give a measure of the discrepancy between two distribution :math:`p(\mathbf{R})` and :math:`\widetilde{q}_\gamma(\mathbf{R})` and is written

.. math::
   \mathcal{D}_{KL}(\widetilde{q}_\gamma \Vert p) = \int d\mathbf{R} \widetilde{q}_\gamma(\mathbf{R}) \ln\bigg(\frac{\widetilde{q}_\gamma(\mathbf{R})}{p(\mathbf{R})}\bigg) \geq 0

In our case, :math:`p(\mathbf{R})` is the Boltzmann weight while :math:`\widetilde{q}_\gamma(\mathbf{R}) = \frac{1}{\widetilde{\mathcal{Z}}} e^{-\beta \widetilde{V}_\gamma(\mathbf{R})}` is a surrogate distribution, associated with a MLIP giving a potential :math:`\widetilde{V}_\gamma(\mathbf{R})`.
An important feature of the surrogate potential is its parametrization, indicated by the subscript :math:`\gamma`, meaning that its shape can be modified by adjusting the parameter :math:`\boldsymbol{\gamma}`.

With some reorganization, the Kullback-Leibler divergence inequality can be reformulated in term of free energies known as the Gibbs-Bogoliubov inequality

.. math::
   \mathcal{F} \leq \widetilde{\mathcal{F}}_\gamma + \langle \mathcal{F} - \widetilde{\mathcal{F}}_\gamma \rangle_\gamma

where :math:`\mathcal{F} = -k_BT \ln(\mathcal{Z})` and :math:`\widetilde{\mathcal{F}}_\gamma = -k_BT \ln(\widetilde{\mathcal{Z}}_\gamma)` are the free energy associated with respectively the true and MLIP potential, and :math:`\langle \rangle_\gamma` indicate an average take with the surrogate potential :math:`\widetilde{V}_\gamma(\mathbf{R})`.

This inequality is at the foundation of the MLACS method.
It indicates that by minimizing its right hand side with respect to the parameters :math:`\boldsymbol{\gamma}` of the surrogate distribution, one can obtain an optimal approximation for the free energy of the system.
Moreover, due to the relation between the Gibbs-Bogoliubov inequality and the Kullback-Leibler divergence, this optimal free energy approximation also correspond to an optimal approximation of the canonical distribution of the system.
Thus, the goal of the MLACS approach is to perform this minimization.

.. image:: pictures/kld.png
   :width: 400
   :alt: Measure of the similarity between two distributions :math:`p(\mathbf{R})` and :math:`\widetilde{q}_\gamma(\mathbf{R})` based on the Kullback–Leibler divergence :math:`\mathcal{D}_{KL}(\widetilde{q}_\gamma \vert p)`
   :align: center

We will assume a potential linearly related to the parameters, with a potential energy written :math:`\widetilde{V}_\gamma(\mathbf{R}) = \sum_n \gamma_n \widetilde{D}_n(\mathbf{R})`.
This correspond to class of widely used MLIP, the most known being SNAP, ACE or MTP.
For such potentials, the minima is obtained by a self-consistent least-squares

.. math::
   \boldsymbol{\gamma} = \langle \widetilde{D}_\gamma(\mathbf{R})^T \widetilde{D}_\gamma(\mathbf{R}) \rangle^{-1} \langle \widetilde{D}_\gamma(\mathbf{R})^T V(\mathbf{R}) \rangle_\gamma

The self consistency comes from the dependence of the optimal parameters to the average in the surrogate ensemble.


Free energy computation
-----------------------

.. image:: pictures/neti.png
   :width: 400
   :align: center
   :alt: valuation of the free energy in two steps: first, using NETI simulations between the “reference system” (the Einstein or Uhlenbeck-Ford model in green) and the “interest system” (the surrogate MLIP potential in yellow), and secondly, using a cumulant expansion between the “interest system” and the ab initio calculation (in blue)

Implementation
**************

.. image:: pictures/workflow_mlacs.png
   :width: 800
   :align: center
