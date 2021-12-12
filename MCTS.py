import numpy as np

from qei_bo import get_gpr_model, optimize_acqf
from Node import Node
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy, UiptCMAESStrategy
from utils import bernoulli, latin_hypercube, from_unit_cube, feature_complementary, ndarray2str, feature_dedup


class MCTS:
    
    def __init__(self, func, dims, lb, ub, feature_batch_size=2, sample_batch_size=3, Cp=0.1, min_num_variables=3, select_right_threshold=5):
        # user defined parameters
        self.func = func
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.feature_batch_size = feature_batch_size
        self.sample_batch_size = sample_batch_size
        self.Cp = Cp
        self.min_num_variables = min_num_variables
        
        self.features = []
        self.samples = []
        self.feature2sample_map = dict()
        self.curt_best_sample = None
        self.curt_best_value = float('-inf')
        self.root_best = -1000
        self.best_value_trace = []
        self.sample_counter = 0
        
        self.split_type = 'mean'
        # self.split_type = 'kmeans'
        # self.uipt_solver = UiptRandomStrategy(self.dims)
        self.uipt_solver = UiptBestKStrategy(self.dims, k=20)
        
        self.nodes = []
        root = Node(parent=None, dims=self.dims, active_dims_idx=list(range(self.dims)), reset_id=True)
        self.nodes.append(root)
        self.ROOT = root
        self.CURT = self.ROOT
        self.num_select_right = 0
        self.select_right_threshold = select_right_threshold
        self.init_train()
        
    def init_train(self):
        assert len(self.features) == 0 and len(self.samples) == 0
        # init features
        features = bernoulli(self.feature_batch_size, self.dims, p=0.5)
        comp_features = [feature_complementary(features[idx]) for idx in range(self.feature_batch_size)]
        self.features.extend( feature_dedup(features + comp_features) )
        
        # collect sample for each feature
        for feature in self.features:
            points = latin_hypercube(self.sample_batch_size, self.dims)
            points = from_unit_cube(points, self.func.lb, self.func.ub)
            for i in range(self.sample_batch_size):
                y = self.func(points[i])
                self.samples.append( (points[i], y) )
                self.update_feature2sample_map(feature, points[i], y)
        
        assert len(self.samples) == len(self.features) * self.sample_batch_size
        
        # update best sample information
        self.sample_counter += len(self.samples)
        X_sample, Y_sample = zip(*self.samples)
        best_sample_idx = np.argmax(Y_sample)
        self.curt_best_sample, self.curt_best_value = self.samples[best_sample_idx]
        self.best_value_trace.append( (self.sample_counter, self.curt_best_value) )
        
        # init
        self.uipt_solver.init_strategy(X_sample, Y_sample)
        
        # print mcts information
        print('='*10)
        print('feature_batch_size: {}'.format(self.feature_batch_size))
        print('sample_batch_size: {}'.format(self.sample_batch_size))
        print('collect {} samples for initializing MCTS'.format(len(self.samples)))
        print('collect {} features for initializing MCTS'.format(len(self.features)))
        print('dims: {}'.format(self.dims))
        print('min_num_variables: {}'.format(self.min_num_variables))
        print('='*10)
        
    def collect_samples(self, feature):
        train_x, train_y = zip(*self.samples)
        np_train_x = np.vstack(train_x)
        np_train_y  = np.array(train_y)
        feature_idx = [idx for idx, i in enumerate(feature) if i == 1]
        ipt_x = np_train_x[:, feature_idx]
        ipt_lb = np.array([i for idx, i in enumerate(self.lb) if idx in feature_idx])
        ipt_ub = np.array([i for idx, i in enumerate(self.ub) if idx in feature_idx])
        # print('select feature: {}'.format(feature))
        
        # get important variables
        gpr = get_gpr_model()
        gpr.fit(ipt_x, train_y)
        new_ipt_x, _ = optimize_acqf(len(feature_idx), gpr, ipt_x, train_y, self.sample_batch_size, ipt_lb, ipt_ub)
        
        # get unimportant variables
        X_sample, Y_sample = [], []
        for i in range(len(new_ipt_x)):
            fixed_variables = {idx: float(v) for idx, v in zip(feature_idx, new_ipt_x[i])}
            new_x = self.uipt_solver.get_full_variable(
                fixed_variables, 
                self.lb, 
                self.ub
            )
            value = self.func(new_x)
            self.uipt_solver.update(new_x, value)
            self.samples.append( (new_x, value) )
            self.update_feature2sample_map(feature, new_x, value)
            
            X_sample.append(new_x)
            Y_sample.append(value)
            
        self.sample_counter += self.sample_batch_size
            
        best_idx = np.argmax(Y_sample)
        if Y_sample[best_idx] > self.curt_best_value:
            self.curt_best_sample = X_sample[best_idx]
            self.curt_best_value = Y_sample[best_idx]
            self.best_value_trace.append( (self.sample_counter, self.curt_best_value) )
            
        return X_sample, Y_sample
    
    def populate_training_data(self):
        self.nodes.clear()
        self.ROOT = Node(parent=None, dims=self.dims, active_dims_idx=list(range(self.dims)), reset_id=True)
        self.nodes.append(self.ROOT)
        self.CURT = self.ROOT
        self.ROOT.init_bag(self.features, self.samples, self.feature2sample_map)
        self.root_best = self.curt_best_value
        assert Node.obj_counter == 1
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
        
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            Y_sample = [y for _, y in node.samples]
            Y_sample = np.array(Y_sample)
            if Y_sample.std() > 0.1:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)
        
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        print('rebuild the tree')
        self.populate_training_data()
        while self.is_splitable():
            to_split = self.get_split_idx()
        
    def greedy_select(self):
        pass
    
    def select(self):
        self.CURT = self.ROOT
        curt_node = self.ROOT
        path = []
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            self.num_select_right += choice
            print('=>', curt_node.get_name(), end=' ')
        print('')
        
        return curt_node, path
    
    def backpropogate(self, leaf, feature, X_sample, Y_sample):
        samples = [(x, y) for x, y in zip(X_sample, Y_sample)]
        best_y = np.max(Y_sample)
        curt_node = leaf
        while curt_node is not None:
            curt_node.n += 1
            # 这里是否需要把值向上传递，因为传递后另一条没有被选择的路径的值并没有被更新，
            # 但似乎不影响，因为计算时只使用了active axis
            curt_node.value += best_y - self.root_best
            curt_node.update_bag(feature, samples)
            curt_node = curt_node.parent
        
    def search(self, iterations):
        for idx in range(iterations):
            print('')
            print('='*10)
            print('iteration: {}'.format(idx))
            print('='*10)
            
            if self.num_select_right >= self.select_right_threshold:
                self.dynamic_treeify()
            leaf, path = self.select()
            # print('='*10)
            # print('iteration: {}'.format(leaf))
            # print('='*10)
            for i in range(1):
                new_feature, new_comp_features = leaf.sample_features(self.feature_batch_size)
                all_features = feature_dedup(new_feature + new_comp_features)
                
                for feature in all_features:
                    if ndarray2str(feature) not in self.feature2sample_map.keys():
                        self.features.append(feature)
                    X_sample, Y_sample = self.collect_samples(feature)
                    self.backpropogate(leaf, feature, X_sample, Y_sample)
                    
            left_kid, right_kid = leaf.split(self.split_type)
            if left_kid is not None and right_kid is not None:
                self.nodes.append(left_kid)
                self.nodes.append(right_kid)
                
            # self.print_tree()
            print('axis_score argsort:', np.argsort(self.ROOT.axis_score))
            print('total samples: {}'.format(len(self.samples)))
            print('current best f(x): {}'.format(self.curt_best_value))
            print('current best x: {}'.format(self.curt_best_sample))
            
    def update_feature2sample_map(self, feature, sample, y):
        feature_str = ndarray2str(feature)
        if self.feature2sample_map.get(feature_str, None) is None:
            self.feature2sample_map[feature_str] = [ (sample, y) ]
        else:
            self.feature2sample_map[feature_str].append( (sample, y) )
            
    def print_tree(self):
        print('='*10)
        for node in self.nodes:
            print('-'*10)
            print(node)
            print('-'*10)
        print('='*10)