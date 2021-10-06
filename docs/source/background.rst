.. _background:
.. index:: Background


Background
==========


Canonical distribution
----------------------


.. math::
    p(\mathbf{R}) = \frac{1}{\mathcal{Z}} e^{-\beta V(\mathbf{R})}


.. math::
    \braket{O(\mathbf{R})} = \frac{1}{\mathcal{Z}} \int \mathrm{d} \mathbf{R} e^{-\beta V(\mathbf{R})} O(\mathbf{R})


Linear Machine-Learning Interatomic Potential
---------------------------------------------

.. math::
    \widetilde{V}(\mathbf{R}) = \widetilde{V}_0 + \sum_k \widetilde{\mathbf{D}}_k(\mathbf{R}) \pmb{\gamma}_k


.. math::
    q(\mathbf{R}) = \frac{1}{\widetilde{\mathcal{Z}}} e^{-\beta \widetilde{V}(\mathbf{R})}


Distribution divergences
------------------------

.. math::
    \mathrm{D}_{\mathrm{KL}} (\widetilde{q} \Vert p) = \int \mathrm{d} \mathbf{R} \widetilde{q}(\mathbf{R}) \mathrm{ln} \bigg[ \frac{\widetilde{q}(\mathbf{R})}{p(\mathbf{R})} \bigg] \geq 0


.. math::
    \nabla_{\pmb{\gamma}} \mathrm{D}_{\mathrm{KL}} (\widetilde{q} \Vert p) = 0

It is possible to show that this minization can be formulated as an equivalent free energy problem

.. math::
    \widetilde{\mathcal{F}} \overset{\mathrm{def}}{=} \widetilde{\mathcal{F}}_0 + \braket{V(\mathbf{R}) - \widetilde{V}(\mathbf{R})}_{\widetilde{V}}

