import numpy as np
from utils import from_unit_cube, latin_hypercube


class CMAES:
    def __init__(self, dims, sigma, pop_size=None, seed=42):
        # user defined input parameters
        self.dims = dims
        self.sigma0 = sigma
        self.sigma = self.sigma0
        self.seed = seed
        
        self.mean = np.random.randn(dims)
        self.mean_old = np.random.randn(dims)
        
        # strategy parameter setting: selection
        self.lam = 4 + np.floor(3*np.log(dims)) if pop_size is None else pop_size
        mu = self.lam / 2
        weights = np.log(mu + 1/2) - np.log(list(range(1, int(mu+1)))) 
        self.mu = np.floor(mu)
        self.weights = weights / np.sum(weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        
        # strategy parameter setting: adaptation
        self.cc = (4 + self.mueff/dims) / (dims + 4 + 2*self.mueff/dims)
        self.cs = (self.mueff + 2) / (dims + self.mueff + 5)
        self.c1 = 2 / ((dims+1.3)**2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1/self.mueff) / ((dims+2)**2 + 2*self.mueff/2)
        self.damps = 1 + 2*np.max((0, np.sqrt((self.mueff-1) / (dims+1)) - 1)) + self.cs
        
        # initialize dynamic strategy parameters and constraints
        self.pc = np.zeros(dims)
        self.ps = np.zeros(dims)
        self.B = np.eye(dims)
        self.D = np.eye(dims)
        self.C = np.eye(dims) # B * D * (B * D)'
        self.chiN = dims**(1/2) * (1 - 1/(4*dims) + 1/(21*dims**2)) # expectation of ||N(0, I)||
        
        self.counteval = 0
        
        print('cma-es init: mu: {}, lambda: {}'.format(self.mu, self.lam))
        
    def ask(self, n=None):
        offspring = []
        offspring_size = self.lam if n is None else n
        for _ in range(int(offspring_size)):
            z = np.random.randn(self.dims)
            x = self.mean + self.sigma * ((self.B @ self.D) @ z)
            offspring.append(x)
        return offspring
    
    def tell(self, population, fitness):
        self.counteval += len(fitness)
        
        child_pop, child_fitness = zip(*sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)) # maximize
        child_pop = child_pop[: int(self.mu)]
        child_fitness = child_fitness[: int(self.mu)]
        
        x = np.vstack(child_pop)
        y = (x - self.mean) / self.sigma
        z = y @ (self.B @ np.linalg.inv(self.D))
        
        self.mean_old = self.mean
        self.mean = self.weights @ x
        # print('check', self.mean.shape)
        zmean = self.weights @ z
        
        self.ps = (1 - self.cs)*self.ps + (np.sqrt(self.cs * (2-self.cs) * self.mueff)) * (self.B @ zmean)
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1-self.cs)**(2*self.counteval/self.lam)) < (1.4 + 2/(self.dims+1))*self.chiN
        self.pc = (1 - self.cc)*self.pc + hsig * np.sqrt(self.cc * (2-self.cc)*self.mueff) * ((self.B@self.D) @ zmean)
        
        # adapt covariance matrix C
        self.C = (1 - self.c1 - self.cmu) * self.C + \
            self.c1 * ((self.pc.reshape(-1, 1) @ self.pc.reshape(1, -1)) + (1-hsig) * self.cc * (2-self.cc) * self.C) + \
            self.cmu * (y.T @ np.diag(self.weights) @ y)
        
        # adapt step-size sigma
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = np.clip(self.sigma, 1e-5, 1)
        
        # update B and D from C
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D, B = np.linalg.eig(self.C)
        self.D = np.diag(np.sqrt(D))
        self.B = B
        
    def conditional_ask(self, n, fixed_variables):
        # calculate p(x1 | x2)
        # fixed_variables: dict, e.g. {0: 1, 2: 3, 5: 10}
        x2_idx, x2 = list(fixed_variables.keys()), list(fixed_variables.values())
        x1_idx = [i for i in range(self.dims) if i not in x2_idx]
        mean1, mean2 = self.mean[x1_idx], self.mean[x2_idx]
        covar11 = self.C[x1_idx][:, x1_idx]
        covar12 = self.C[x1_idx][:, x2_idx]
        covar21 = self.C[x2_idx][:, x1_idx]
        covar22 = self.C[x2_idx][:, x2_idx]
        
        inv_covar22 = np.linalg.inv(covar22)
        cond_mean = mean1 + covar12 @ inv_covar22 @ (x2 - mean2)
        cond_covar = covar11 - covar12 @ inv_covar22 @ covar21
        
        cond_covar = np.triu(cond_covar) + np.triu(cond_covar).T
        # D, B = np.linalg.eig(cond_covar)
        # D = np.diag(np.sqrt(D))
        
        offspring = []
        for _ in range(n):
            # z = np.random.randn(len(x1_idx))
            # cond_x = cond_mean + ((B @ D) @ z)
            
            cond_x = np.random.multivariate_normal(np.real(cond_mean), np.real(cond_covar))
            # print('cond mean:', cond_mean)
            # print('cond covar:', cond_covar)
            # print('covar:', self.C)
            x = np.zeros(self.dims)
            x[x1_idx] = cond_x
            x[x2_idx] = x2
            offspring.append(x)
        
        return offspring
    

if __name__ == '__main__':
    from benchmark_problems import levy10_10
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='levy10_10', type=str, choices=('levy10_10', ))
    parser.add_argument('--iterations', default=100, type=int)
    args = parser.parse_args()
    
    func = eval(args.func)
    cmaes = CMAES(func.dims, 0.01 / np.sqrt(func.dims))
    fitness_list = []
    best_eval = []
    for _ in range(args.iterations):
        pop = cmaes.ask()
        fitness = [func(p) for p in pop]
        cmaes.tell(pop, fitness)
        fitness_list.extend(fitness)
        best_eval.append(max(fitness_list))
        print('covar:', cmaes.C)
    
    print('final function value: {}, the number of samples: {}'.format(best_eval[-1], args.iterations*cmaes.lam))
    plt.plot(best_eval)
    plt.show()
    