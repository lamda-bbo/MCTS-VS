import numpy as np
from nasbench import api


NASBENCH_TFRECORD = '/dataset/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2 # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]


class NasBench:
    def __init__(self, path=NASBENCH_TFRECORD, seed=None):
        self.nasbench = api.NASBench(path, seed)
        self.dims = 36 # 15 + 21
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)
        self.opt_val = 1.0
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        ops_param = [x[i*3: (i+1)*3]for i in range(5)]
        edge_param = x[-21: ]
        
        # layer type
        ops = [INPUT] + [ALLOWED_OPS[np.argmax(ops_param[i])] for i in range(5)] + [OUTPUT]
        
        # edge
        mat_param = np.zeros((7, 7))
        cur_idx = 0
        for i in range(7):
            mat_param[i][i+1: ] = edge_param[cur_idx: cur_idx+7-i-1]
            cur_idx += 7-i-1
        assert np.all(np.triu(mat_param) == mat_param)
        
        mat = np.zeros((7, 7))
        result = 0
        while True:
            max_idx = np.argmax(mat_param)
            row, col = int(max_idx / 7), int(max_idx % 7)
            if mat_param[row][col] > 1e-5:
                mat[row][col] = 1
                mat_param[row][col] = 0
                cell = api.ModelSpec(matrix=mat.astype(int), ops=ops)
                try:
                    data = self.nasbench.query(cell)
                    train_min = data['training_time'] / 60
                    if train_min < 30:
                        result = data['test_accuracy']
                except api.OutOfDomainError as e:
                    e = str(e)
                    if e == 'invalid spec, provided graph is disconnected.':
                        pass
                    elif e.startswith('too many edges'):
                        mat[row][col] = 0
                        break
                    else:
                        assert 0, e
            else:
                break
        # print(ops_param)
        # print(ops)
        # print(mat)
                
        return result
            
        
if __name__ == '__main__':
    nas_problem = NasBench()
    result = nas_problem(np.random.uniform(0, 1, 36))
    print(result)