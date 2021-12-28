import numpy as np
import pandas as pd
import os


def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


def bernoulli(n, dims, p=0.5):
    assert n <= 2**dims, 'the number of init samples is larger than the whole space'
    if n > 2**(dims-1):
        print('too many init samples')
        
    points = []
    i = 0
    while i < n:
        point = np.zeros(dims)
        prob = np.random.uniform(0.0, 1.0, dims)
        point[prob < p] = 1
        point = list(point)
        if point not in points:
            points.append(point)
            i += 1
    
    points = [np.array(point) for point in points]
    
    # points = np.zeros((n, dims))
    # prob = np.random.uniform(0.0, 1.0, (n, dims))
    # points[prob < p] = 1
    
    return points


def feature_complementary(feature):
    comp = []
    for i in feature:
        i_comp = 0 if i else 1
        comp.append(i_comp)
    return np.array(comp)


def ndarray2str(arr):
    s = ''
    for i in arr:
        s += str(int(i))
    return s


def pad_str_to_8chars(ins, total):
    if len(ins) <= total:
        ins += ' '*(total - len(ins) )
        return ins
    else:
        return ins[0:total]
    

def feature_dedup(features):
    feature_set = set()
    dedup = []
    for f in features:
        feature_set.add(tuple(f))
    for f in feature_set:
        dedup.append(np.array(f))
    return dedup


def save_results(root_dir, algo, func, seed, df_data):
    os.makedirs(root_dir, exist_ok=True)
    save_dir = os.path.join(root_dir, func)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '%s-%d.csv' % (algo, seed))
    df_data.to_csv(save_path)
    print('save %s result into: %s' % (algo, save_path))
    
    
def save_args(root_dir, algo, func, seed, args):
    os.makedirs(root_dir, exist_ok=True)
    save_dir = os.path.join(root_dir, func)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '%s-%d.txt' % (algo, seed))
    with open(save_path, 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))
    print('save {} config into: {}'.format(algo, save_path))
    

if __name__ == '__main__':
    # print(latin_hypercube(3, 6))
    # print(bernoulli(3, 3))
    features = [np.array([1, 0]), np.array([1, 0]), np.array([1, 1])]
    print(feature_dedup(features))
    