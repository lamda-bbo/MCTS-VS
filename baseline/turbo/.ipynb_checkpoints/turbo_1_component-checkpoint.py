import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube
from .turbo_1 import Turbo1


class Turbo1_Component(Turbo1):
    def optimize(self, X_init=None, fX_init=None, n=1):
        cnt = 0
        while self.n_evals < self.max_evals and cnt < n:
            cnt += 1
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()
            
            # Generate and evalute initial design points
            if X_init is None:
                X_init = latin_hypercube(self.n_init, self.dim)
                X_init = from_unit_cube(X_init, self.lb, self.ub)
            if fX_init is None:
                fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))
                
        return self.X, self.fX
                
        
class Turbo1_VS_Component(Turbo1):
    def optimize(self, X_init, fX_init, feature_idx, uipt_solver, n=1):
        fX_init = fX_init.reshape(-1, 1)
        cnt = 0
        self.n_evals = 0
        X_sample, Y_sample = [], []
        while self.n_evals < self.max_evals and cnt < n:
            cnt += 1
            # Initialize parameters
            self._restart()
            
            # Update budget and set as initial data for this TR
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)
                
                # Evaluate batch
                fX_next = []
                for i in range(len(X_next)):
                    fixed_variables = {idx: float(v) for idx, v in zip(feature_idx, X_next[i])}
                    new_x = uipt_solver.get_full_variable(
                        fixed_variables, 
                        self.lb, 
                        self.ub
                    )
                    value = self.f(new_x)
                    fX_next.append([value])
                    
                    # update global store
                    X_sample.append(new_x)
                    Y_sample.append(value)
                    uipt_solver.update(new_x, -value)

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

        return X_sample, Y_sample
