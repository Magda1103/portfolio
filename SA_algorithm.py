import random
import math
import time

SAFE_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "sin": math.sin,
    "cos": math.cos,
    "cosh": math.cosh,
    "exp": math.exp,
    "log": math.log,
    "pi": math.pi,
    "e": math.e,
    "pow": pow,
    "sqrt": math.sqrt,
    "max": max,
    "abs": abs,
}



class SA:

    def __init__(self, user_func_str, variable_names, T0, alpha, k, M, domain, mode='min'):
        self.T0 = T0
        self.alpha = alpha
        self.k = k
        self.M = M
        self.domain = domain
        self.mode = mode

        lambda_vars = ", ".join(variable_names)

        try:
            self.f = eval(f"lambda {lambda_vars}: {user_func_str}", SAFE_GLOBALS, {})
            test_args = [0] * len(variable_names)
            self.f(*test_args)
        except Exception as e:
            raise ValueError(f"Błąd podczas przetwarzania wzoru '{user_func_str}': {e}")

    def algorithm_SA(self, save_history=False):
        num_dimensions = len(self.domain)

        s = [random.uniform(d[0], d[1]) for d in self.domain]
        T = self.T0
        f = self.f

        start_time = time.time()


        best_s = list(s)
        best_fs = f(*s)

        history = []

        for i in range(self.M):
            fs = f(*s)


            s_prim = []
            for j in range(num_dimensions):
                low, high = self.domain[j]
                step = random.uniform(-T, T)
                new_val = s[j] + step
                new_val = max(min(new_val, high), low)
                s_prim.append(new_val)

            fs_prim = f(*s_prim)
            delta = fs_prim - fs



            if self.mode == 'min':
                if delta < 0:
                    s = s_prim

                    if fs_prim < best_fs:
                        best_s = s_prim
                        best_fs = fs_prim
                        if save_history:
                            history.append((i, best_s.copy(), best_fs))
                else:
                    rand_check = random.uniform(0, 1)
                    if rand_check < math.exp(-delta / (self.k * T)):
                        s = s_prim

            else:
                if delta > 0:
                    s = s_prim

                    if fs_prim > best_fs:
                        best_s = s_prim
                        best_fs = fs_prim
                        if save_history:
                            history.append((i, best_s.copy(), best_fs))
                else:
                    rand_check = random.uniform(0, 1)
                    if rand_check < math.exp(delta / (self.k * T)):
                        s = s_prim


            T *= self.alpha
        end_time = time.time()
        elapsed_time = end_time - start_time

        if save_history:
            return best_s, best_fs, history, elapsed_time
        return best_s, best_fs, elapsed_time

