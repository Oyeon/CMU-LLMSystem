from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph
    starting from `variable`. Returns a list [leaf_node, ..., variable].
    """
    visited = set()
    topo_order = []

    def dfs(node: Variable) -> None:
        # If we've visited or the node is a constant, skip
        # so we don't re-visit constants or re-visit nodes
        if (node.unique_id in visited) or node.is_constant():
            return
        visited.add(node.unique_id)
        # Visit each parent (dependency) first (DFS post-order)
        for parent in node.parents:
            dfs(parent)
        # Post-order: add the node after visiting children
        topo_order.append(node)

    dfs(variable)
    # topo_order is [leaves -> variable], but for backprop
    # we usually want [variable -> leaves]. So reverse it:
    return reversed(topo_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for all leaf nodes. 
    `deriv` is usually 1.0 if `variable` is a scalar loss.
    """
    # 1. Get topological ordering from variable to leaves
    order = list(topological_sort(variable))
    # 2. A map from variable's unique_id to its accumulated gradient
    grads = {}
    # 3. The "final" node's gradient is the user-supplied `deriv`
    grads[variable.unique_id] = deriv

    # 4. Traverse in topological order, from final node -> leaves
    for v in order:
        d_out = grads.get(v.unique_id, 0.0)
        if v.is_leaf():
            # If it's a leaf node, just accumulate its gradient
            v.accumulate_derivative(d_out)
        else:
            # If it's not a leaf, propagate gradients to its parents
            for (parent, parent_grad_contrib) in v.chain_rule(d_out):
                if not parent.is_constant():
                    grads[parent.unique_id] = grads.get(parent.unique_id, 0.0) + parent_grad_contrib


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
