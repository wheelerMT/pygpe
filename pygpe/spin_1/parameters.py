class Parameters:

    def __init__(self, parameters: dict) -> None:
        """Constructs a class that contains all the parameters needed for the spin-1 evolution.

        :param parameters: A dictionary containing all the parameters of the system.
        """
        self.c0 = parameters["spin_indep_int"]
        self.c2 = parameters["spin_dep_int"]
        self.p = parameters["linear_zeeman"]
        self.q = parameters["quadratic_zeeman"]
        self.trap = parameters["trap_pot"]

        # Time-related params
        self.dt = parameters["time_step"]
        self.nt = parameters["num_time_steps"]
