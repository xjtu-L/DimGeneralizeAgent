"""Custom Reifier for subgraph-level symbolic dimensions.

Handles the diverse shape patterns found in lt_submit4.7 subgraphs,
which differ from the whole-model patterns in GraphNet's built-in reifiers.
"""

from graph_net.torch.sym_dim_reifiers.reify_util import get_dynamic_dim_constraints
from graph_net.torch.sym_dim_reifiers.reifier_base import ReifierBase
import sympy


class ConcreteReifier(ReifierBase):
    """Reifier for subgraph-level computation graphs.

    Maps symbolic dimensions to concrete values based on position semantics:
    - S0, S1 (first 2 symbols) -> (batch, seq_len) for NLP
    - S0 (single symbol) -> varies by context
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.dyn_dim_cstrs = get_dynamic_dim_constraints(model_path)

    def get_reifier_name(self) -> str:
        return "subgraph_sym_dim_reifier"

    def match(self) -> bool:
        if self.dyn_dim_cstrs is None:
            return False
        sym_shapes = self.dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()
        return sym_shapes != "[]" and len(self.dyn_dim_cstrs.symbols) > 0

    def reify(self):
        assert self.match()
        symbols = self.dyn_dim_cstrs.symbols
        num_symbols = len(symbols)

        if num_symbols == 1:
            # Single symbolic dimension - typically seq_len
            return self._reify_single_symbol(symbols)
        elif num_symbols == 2:
            # Two symbolic dimensions - typically (batch, seq_len)
            return self._reify_two_symbols(symbols)
        else:
            # More symbols - use generalized mapping
            return self._reify_multi_symbols(symbols)

    def _reify_single_symbol(self, symbols):
        """Single symbolic dim (S0) -> seq_len variations."""
        S0 = symbols[0]
        # Check if it looks like a batch dim (example_value == 1) or seq_len
        ev = self.dyn_dim_cstrs.symbol2example_value.get(S0, 4)

        if ev <= 1:
            # Batch dimension
            return {S0: [1, 1, 16, 32, 8, 4, 2, 64, 128]}
        else:
            # Seq_len or other dimension
            return {S0: [64, 512, 128, 64, 256, 512, 1024, 128, 64]}

    def _reify_two_symbols(self, symbols):
        """Two symbolic dims (S0, S1) -> (batch, seq_len) pairs."""
        S0, S1 = symbols[0], symbols[1]
        return {
            (S0, S1): [
                [1, 64],
                [1, 512],
                [16, 128],
                [32, 64],
                [8, 256],
                [4, 512],
                [2, 1024],
                [64, 128],
                [128, 64],
            ]
        }

    def _reify_multi_symbols(self, symbols):
        """Multiple symbolic dims -> use first two as (batch, seq_len), rest as constants."""
        S0, S1 = symbols[0], symbols[1]
        base = self._reify_two_symbols(symbols[:2])
        result = {}

        for key, values in base.items():
            extended_key = symbols
            extended_values = []
            for pair in values:
                extended = list(pair)
                # For additional symbols, keep their example values
                for s in symbols[2:]:
                    ev = self.dyn_dim_cstrs.symbol2example_value.get(s, 1)
                    extended.append(ev)
                extended_values.append(extended)
            result[tuple(extended_key)] = extended_values

        return result
