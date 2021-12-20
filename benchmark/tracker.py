import pandas as pd
import os
import time


class Tracker:
    def __init__(self, save_interval, save_config):
        self.counter = 0
        self.best_value_trace = []
        self.curt_best = float('-inf')
        self.save_interval = save_interval
        self.save_config = save_config
        self.start_time = time.time()
        
    def track(self, result):
        self.counter += 1
        if result > self.curt_best:
            self.curt_best = result
        self.best_value_trace.append((
            self.counter, 
            self.curt_best,
            time.time() - self.start_time
        ))
        
        if self.counter % self.save_interval == 0:
            df_data = pd.DataFrame(self.best_value_trace, columns=['x', 'y', 't'])
            save_results(
                self.save_config['root_dir'],
                self.save_config['algo'],
                self.save_config['func'],
                self.save_config['seed'],
                df_data,
            )
        
        
def save_results(root_dir, algo, func, seed, df_data):
    os.makedirs(root_dir, exist_ok=True)
    save_dir = os.path.join(root_dir, func)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '%s-%d.csv' % (algo, seed))
    df_data.to_csv(save_path)
    print('save %s result into: %s' % (algo, save_path))