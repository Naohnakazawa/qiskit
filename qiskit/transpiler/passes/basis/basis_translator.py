# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Translates gates to a target basis using a given equivalence library."""

import logging

from collections import defaultdict

from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.basis_translator import base_run

logger = logging.getLogger(__name__)


class BasisTranslator(TransformationPass):
    """Translates gates to a target basis by searching for a set of translations
    from a given EquivalenceLibrary.

    This pass operates in several steps:

    * Determine the source basis from the input circuit.
    * Perform a Dijkstra search over basis sets, starting from the device's
      target_basis new gates are being generated using the rules from the provided
      EquivalenceLibrary and the search stops if all gates in the source basis have
      been generated.
    * The found path, as a set of rules from the EquivalenceLibrary, is composed
      into a set of gate replacement rules.
    * The composed replacement rules are applied in-place to each op node which
      is not already in the target_basis.

    If the target keyword argument is specified and that
    :class:`~qiskit.transpiler.Target` objects contains operations
    which are non-global (i.e. they are defined only for a subset of qubits),
    as calculated by :meth:`~qiskit.transpiler.Target.get_non_global_operation_names`,
    this pass will attempt to match the output translation to those constraints.
    For 1 qubit operations this is straightforward, the pass will perform a
    search using the union of the set of global operations with the set of operations
    defined solely on that qubit. For multi-qubit gates this is a bit more involved,
    while the behavior is initially similar to the single qubit case, just using all
    the qubits the operation is run on (where order is not significant) isn't sufficient.
    We also need to consider any potential local qubits defined on subsets of the
    quantum arguments for the multi-qubit operation. This means the target used for the
    search of a non-global multi-qubit gate is the union of global operations, non-global
    multi-qubit gates sharing the same qubits, and any non-global gates defined on
    any subset of the qubits used.


    .. note::

        In the case of non-global operations it is possible for a single
        execution of this pass to output an incomplete translation if any
        non-global gates are defined on qubits that are a subset of a larger
        multi-qubit gate. For example, if you have a ``u`` gate only defined on
        qubit 0 and an ``x`` gate only on qubit 1 it is possible when
        translating a 2 qubit operation on qubit 0 and 1 that the output might
        have ``u`` on qubit 1 and ``x`` on qubit 0. Typically running this pass
        a second time will correct these issues.

    .. _translation_errors:

    Translation Errors
    ------------------

    This pass will error if there is no path to translate an input gate to
    the specified basis. However, during a typical/default preset passmanager
    this pass gets run multiple times at different stages of the compilation
    pipeline. This means that potentially the input gates that are getting
    translated were not in the input circuit to :func:`~.transpile` as they
    were generated by an intermediate transform in the circuit.

    When this error occurs it typically means that either the target basis
    is not universal or there are additional equivalence rules needed in the
    :class:`~.EquivalenceLibrary` instance being used by the
    :class:`~.BasisTranslator` pass. You can refer to
    :ref:`custom_basis_gates` for details on adding custom equivalence rules.
    """

    def __init__(self, equivalence_library, target_basis, target=None, min_qubits=0):
        """Initialize a BasisTranslator instance.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            target_basis (list[str]): Target basis names to unroll to, e.g. ``['u3', 'cx']``.
            target (Target): The backend compilation target
            min_qubits (int): The minimum number of qubits for operations in the input
                dag to translate.
        """
        super().__init__()
        self._equiv_lib = equivalence_library
        self._target_basis = target_basis
        # Bypass target if it doesn't contain any basis gates (i.e. it's a _FakeTarget), as this
        # not part of the official target model.
        self._target = target if target is not None and len(target.operation_names) > 0 else None
        self._non_global_operations = None
        self._qargs_with_non_global_operation = {}
        self._min_qubits = min_qubits
        if self._target is not None:
            self._non_global_operations = self._target.get_non_global_operation_names()
            self._qargs_with_non_global_operation = defaultdict(set)
            for gate in self._non_global_operations:
                for qarg in self._target[gate]:
                    self._qargs_with_non_global_operation[qarg].add(gate)

    def run(self, dag):
        """Translate an input DAGCircuit to the target basis.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            TranspilerError: if the target basis cannot be reached

        Returns:
            DAGCircuit: translated circuit.
        """

        out = base_run(
            dag,
            self._equiv_lib,
            self._qargs_with_non_global_operation,
            self._min_qubits,
            None if self._target_basis is None else set(self._target_basis),
            self._target,
            None if self._non_global_operations is None else set(self._non_global_operations),
        )
        # If Rust-space basis translation returns `None`, it's because the input DAG is already
        # suitable and it didn't need to modify anything.
        return dag if out is None else out
