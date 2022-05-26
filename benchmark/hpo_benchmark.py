import numpy as np
from hpobench.container.benchmarks.nas.nasbench_1shot1 import NASBench1shot1SearchSpace1Benchmark
from hpobench.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark, Cifar100NasBench201Benchmark, ImageNetNasBench201BenchmarkOriginal, Cifar100NasBench201BenchmarkOriginal
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class HpoNasBench:
    def __init__(self, name, seed=None):
        if name == 'nasbench201':
            self.benchmark = ImageNetNasBench201BenchmarkOriginal()
            # self.benchmark = Cifar100NasBench201Benchmark()
        elif name == 'nasbench1shot1':
            self.benchmark = NASBench1shot1SearchSpace1Benchmark()
        else:
            assert 0
        
        self.name = name
        self.config_space = self.benchmark.get_configuration_space()
        print('config_space:')
        print('=============')
        print(self.config_space)
        print('=============')
        
        self.hp = self.config_space.get_hyperparameters()
        self.name_list = [i.name for i in self.hp]
        self.choices_list = [i.choices for i in self.hp]
        self.n_category = [len(i) for i in self.choices_list]
        
        self.dims = sum(self.n_category)
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)
        self.opt_val = 1.0
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        i = 0
        values = dict()
        for idx, j in enumerate(self.n_category):
            choice = np.argmax(x[i: i+j])
            values[self.name_list[idx]] = self.choices_list[idx][choice]
            i += j
        config = CS.Configuration(self.config_space, values=values)
        print(config)
        result_dict = self.benchmark.objective_function(configuration=config)
        # print(result_dict)
        if self.name == 'nasbench201':
            print(result_dict['info'])
            result = result_dict['info']['valid_precision'] / 100
        elif self.name == 'nasbench1shot1':
            result = np.mean(result_dict['info']['test_accuracies'])
        return result


if __name__ == '__main__':
    # nas_problem = HpoNasBench('nasbench1shot1')
    nas_problem = HpoNasBench('nasbench1shot1')
    print('dims:', nas_problem.dims)
    acc = nas_problem(np.random.uniform(0, 1, nas_problem.dims))
    print(acc)