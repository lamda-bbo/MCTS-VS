from abc import ABCMeta
import numpy as np
# from pycmaes import CMAES


class UiptStrategy(metaclass=ABCMeta):
    def __init__(self, dims, seed=42):
        self.dims = dims
        self.seed = seed
        
    def init_strategy(self, xs, ys):
        for x, y in zip(xs, ys):
            self.update(x, y)
    
    def get_full_variable(self, fixed_variables, lb, ub):
        pass
    
    def update(self, x, y):
        pass


class UiptRandomStrategy(UiptStrategy):
    def __init__(self, dims, seed=42):
        UiptStrategy.__init__(self, dims, seed)
        
    def get_full_variable(self, fixed_variables, lb, ub):
        new_x = np.zeros(self.dims)
        for dim in range(self.dims):
            if dim in fixed_variables.keys():
                new_x[dim] = fixed_variables[dim]
            else:
                new_x[dim] = np.random.uniform(lb[dim], ub[dim])
        return new_x
    

class UiptBestKStrategy(UiptStrategy):
    def __init__(self, dims, k=10, seed=42):
        UiptStrategy.__init__(self, dims, seed)
        self.k = k
        self.best_xs = []
        self.best_ys = []
    
    def get_full_variable(self, fixed_variables, lb, ub):
        best_xs = np.asarray(self.best_xs)
        best_ys = np.asarray(self.best_ys)
        new_x = np.zeros(self.dims)
        for dim in range(self.dims):
            if dim in fixed_variables.keys():
                new_x[dim] = fixed_variables[dim]
            else:
                new_x[dim] = np.random.choice(best_xs[:, dim])
        return new_x
    
    def update(self, x, y):
        if len(self.best_xs) < self.k:
            self.best_xs.append(x)
            self.best_ys.append(y)
            if len(self.best_xs) == self.k:
                self.best_xs = np.vstack(self.best_xs)
                self.best_ys = np.array(self.best_ys)
        else:
            min_y = np.min(self.best_ys)
            if y > min_y:
                idx = np.random.choice(np.argwhere(self.best_ys == min_y).reshape(-1))
                self.best_xs[idx] = x
                self.best_ys[idx] = y
        assert len(self.best_xs) <= self.k
        
        
class UiptAverageBestKStrategy(UiptStrategy):
    def __init__(self, dims, k=10, seed=42):
        UiptStrategy.__init__(self, dims, seed)
        self.k = k
        self.best_xs = []
        self.best_ys = []
    
    def get_full_variable(self, fixed_variables, lb, ub):
        best_xs = np.asarray(self.best_xs)
        best_ys = np.asarray(self.best_ys)
        average_best_x = best_xs.mean(axis=0)
        
        new_x = np.zeros(self.dims)
        for dim in range(self.dims):
            if dim in fixed_variables.keys():
                new_x[dim] = fixed_variables[dim]
            else:
                new_x[dim] = average_best_x[dim]
        return new_x
    
    def update(self, x, y):
        if len(self.best_xs) < self.k:
            self.best_xs.append(x)
            self.best_ys.append(y)
            if len(self.best_xs) == self.k:
                self.best_xs = np.vstack(self.best_xs)
                self.best_ys = np.array(self.best_ys)
        else:
            min_y = np.min(self.best_ys)
            if y > min_y:
                idx = np.random.choice(np.argwhere(self.best_ys == min_y).reshape(-1))
                self.best_xs[idx] = x
                self.best_ys[idx] = y
        assert len(self.best_xs) <= self.k
        
        
class UiptCopyStrategy(UiptStrategy):
    def __init__(self, dims, seed=42):
        self.copy_strategy = UiptBestKStrategy(dims, 1, seed)
    
    def get_full_variable(self, fixed_variables, lb, ub):
        return self.copy_strategy.get_full_variable(fixed_variables, lb, ub)
    
    def update(self, x, y):
        self.copy_strategy.update(x, y)
        
        
class UiptMixStrategy(UiptStrategy):
    def __init__(self, dims, p=0.1, seed=42):
        self.p = p
        self.random_strategy = UiptRandomStrategy(dims, seed)
        self.copy_strategy = UiptCopyStrategy(dims, seed)
        
    def get_full_variable(self, fixed_variables, lb, ub):
        if np.random.uniform() < self.p:
            return self.random_strategy.get_full_variable(fixed_variables, lb, ub)
        else:
            return self.copy_strategy.get_full_variable(fixed_variables, lb, ub)
    
    def update(self, x, y):
        self.random_strategy.update(x, y)
        self.copy_strategy.update(x, y)
