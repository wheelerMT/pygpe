import cupy as cp


class Parameters:

    def __init__(self, spin_indep_int: float, spin_dep_int: float, linear_zeeman: float, quadratic_zeeman: float,
                 trap_pot: cp.ndarray | float = 0.) -> None:
        """Constructs a class that contains all the parameters needed for the spin-1 evolution.

        :param spin_indep_int: The spin independent interaction strength, c_0.
        :param spin_dep_int: The spin dependent interaction strength, c_2.
        :param linear_zeeman: The linear Zeeman energy.
        :param quadratic_zeeman: The quadratic Zeeman energy.
        :param trap_pot: The trapping potential of the system.
        """
        self.c0 = spin_indep_int
        self.c2 = spin_dep_int
        self.p = linear_zeeman
        self.q = quadratic_zeeman
        self.trap = trap_pot
