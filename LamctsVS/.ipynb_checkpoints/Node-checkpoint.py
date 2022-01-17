import numpy as np
from sklearn.cluster import KMeans
from utils import ndarray2str, pad_str_to_8chars


class Node:
    obj_counter = 0
    
    def __init__(self, parent=None, dims=0, active_dims_idx=None, min_num_variables=3, reset_id=False):
        # every node is initialized as a leaf
        self.dims = dims
        self.active_dims_idx = active_dims_idx
        self.min_num_variables = min_num_variables
        self.value = 0
        self.n = 0
        self.uct = 0
        self.kmeans = KMeans(n_clusters=2)
        
        self.parent = parent
        self.kids = [] # 0: good, 1: bad
        self.features, self.samples = [], []
        self.feature2sample_map = dict()
        
        if reset_id:
            Node.obj_counter = 0
        
        self.id = Node.obj_counter
        Node.obj_counter += 1
        
    def init_bag(self, features, samples, feature2sample_map):
        # clear and init
        self.features.clear()
        self.samples.clear()
        self.feature2sample_map = dict()
        
        self.features.extend(features)
        self.samples.extend(samples)
        
        for k, v in feature2sample_map.items():
            self.feature2sample_map[k] = []
            self.feature2sample_map[k].extend(v)
        
    def update_bag(self, feature, samples):
        self.samples.extend(samples)
        k = ndarray2str(feature)
        if self.feature2sample_map.get(k, None) is None:
            self.features.append(feature)
            self.feature2sample_map[k] = []
        self.feature2sample_map[k].extend(samples)
    
    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.active_axis_score.shape[0]
        zero_label_score = []
        one_label_score = []
        for idx in range(len(plabel)):
            if plabel[idx] == 0:
                zero_label_score.append(self.active_axis_score[idx])
            elif plabel[idx] == 1:
                one_label_score.append(self.active_axis_score[idx])
            else:
                assert 0
        good_label_mean = np.mean(zero_label_score)
        bad_label_mean = np.mean(one_label_score)
        return good_label_mean, bad_label_mean
    
    def split(self, split_type='median'):
        if len(self.active_dims_idx) < self.min_num_variables:
            return None, None
        # if self.active_axis_score.var() < 0.1:
        #     return None, None
        
        if split_type in ['median', 'mean']:
            threshold = getattr(self, split_type) # calculated by active dims
            good_idx = np.where(self.axis_score >= threshold)[0]
            good_idx = sorted(set(list(good_idx)) & set(list(self.active_dims_idx)))
            bad_idx = np.where(self.axis_score < threshold)[0]
            bad_idx = sorted(set(list(bad_idx)) & set(list(self.active_dims_idx)))
        elif split_type == 'kmeans':
            # 0: good cluster, 1: bad cluster
            self.kmeans = self.kmeans.fit(self.active_axis_score.reshape(-1, 1))
            plabel = self.kmeans.predict(self.active_axis_score.reshape(-1, 1))
            good_label_mean, bad_label_mean = self.get_cluster_mean(plabel)
            
            if good_label_mean < bad_label_mean:
                for idx in range(len(plabel)):
                    if plabel[idx] == 0:
                        plabel[idx] = 1
                    else:
                        plabel[idx] = 0
                        
            good_idx, bad_idx = [], []
            for idx, label in zip(self.active_dims_idx, plabel):
                if label == 0:
                    good_idx.append(idx)
                else:
                    bad_idx.append(idx)
        else:
            assert 0
        
        assert ( set(good_idx) | set(bad_idx) == set(list(self.active_dims_idx)) )
        assert ( len(set(good_idx) & set(bad_idx)) == 0 )
        
        if len(good_idx) < self.min_num_variables and len(bad_idx) < self.min_num_variables:
            return None, None
        
        if len(good_idx) == 0 or len(bad_idx) == 0:
            return None, None
        
        left = Node(parent=self, dims=self.dims, active_dims_idx=good_idx, min_num_variables=self.min_num_variables, reset_id=False)
        left.init_bag(self.features, self.samples, self.feature2sample_map)
        right = Node(parent=self, dims=self.dims, active_dims_idx=bad_idx, min_num_variables=self.min_num_variables, reset_id=False)
        right.init_bag(self.features, self.samples, self.feature2sample_map)
        
        self.kids = [left, right]
        return left, right
    
    def get_axis_cnt(self):
        axis_cnt = np.zeros(self.dims)
        for feature in self.features:
            feature_str = ndarray2str(feature)
            axis_cnt += feature * len(self.feature2sample_map[feature_str])
        return axis_cnt
        
    def get_axis_score(self):
        axis_cnt = np.zeros(self.dims)
        axis_score = np.zeros(self.dims)
        
        for feature in self.features:
            feature_str = ndarray2str(feature)
            axis_cnt += feature * len(self.feature2sample_map[feature_str])
            score = np.max([y for _, y in self.feature2sample_map[feature_str]])
            axis_score += score * feature
        
        axis_score /= (axis_cnt + 1e-6)
        return axis_score
    
    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False
        
    def get_uct(self, Cp):
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        return self.max + 2 * Cp * np.sqrt(2 * np.power(self.parent.n, 0.5) / self.n)
    
    def get_name(self):
        return 'node' + str(self.id)
    
    def sample_features(self, n=1, p=0.5):
        features = []
        comp_features = []
        for _ in range(n):
            feature = np.zeros(self.dims)
            comp_feature = np.zeros(self.dims)
            
            active_prob = np.random.uniform(0.0, 1.0, len(self.active_dims_idx))
            
            active_feature = np.zeros(len(self.active_dims_idx))
            active_feature[active_prob < p] = 1
            feature[self.active_dims_idx] = active_feature
            active_comp_feature = np.ones(len(self.active_dims_idx))
            active_comp_feature[active_prob < p] = 0
            comp_feature[self.active_dims_idx] = active_comp_feature
            
            if np.sum(feature) == 0:
                active_idx = np.random.choice(self.active_dims_idx)
                feature[active_idx] = 1
                comp_feature[active_idx] = 0
            elif np.sum(feature) == len(self.active_dims_idx):
                active_idx = np.random.choice(self.active_dims_idx)
                feature[active_idx] = 0
                comp_feature[active_idx] = 1
            
            if np.sum(feature) != 0:
                features.append(feature)
            if np.sum(comp_feature) != 0:
                comp_features.append(comp_feature)
        return features, comp_features

    @property
    def axis_score(self):
        return self.get_axis_score()
    
    @property
    def active_axis_score(self):
        return self.axis_score[self.active_dims_idx]
    
    @property
    def mean(self):
        return np.mean(self.active_axis_score)
    
    @property
    def median(self):
        return np.median(self.active_axis_score)
        
    @property
    def max(self):
        return np.max(self.active_axis_score)
    
    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False
    
    def __str__(self):
        name = self.get_name()
        name = pad_str_to_8chars(name, 7)
        
        name += 'mean: {}, uct: {}  '.format(self.mean, self.get_uct(0.1))
        
        name += ( pad_str_to_8chars( 'is good:' + str(self.is_good_kid() ), 15 ) )
        
        parent = '---'
        if self.parent is not None:
            parent = self.parent.get_name()
        name += ('parent:' + pad_str_to_8chars(parent, 10))
        
        kids = ''
        kid  = ''
        for k in self.kids:
            kid   = pad_str_to_8chars( k.get_name(), 10 )
            kids += kid
        name += (' kids:' + kids)
        
        active_dim_idx_str = ' '.join([str(i) for i in self.active_dims_idx])
        name += ('\n  active dim idx: ' + active_dim_idx_str)
        axis_cnt_str = ' '.join([str(i) for i in self.get_axis_cnt()[self.active_dims_idx]])
        name += ('\n  axis cnt: ' + axis_cnt_str)
        active_axis_score_str = ' '.join(['%.2f'%(i) for i in self.active_axis_score])
        name += ('\n  active axis score: ' + active_axis_score_str)
        
        return name