import pandas as pd


class DummyModel:
    def __init__(self, **kwargs):
        self._index = 0
        pass

    def compile_model(self, **kwargs):
        pass
    
    def fit_model(self, **kwargs):      
        pass
    
    def predict(self, frames, **kwargs):        
        pred = self._test_df.loc[self._index].tolist()
        self._index += 1
        return pred

    def predict_from_saved(self, test_image_dir, 
                           frames_file,
                           **kwargs):      
        pred = self._test_df.loc[self._index].tolist()
        self._index += 1
        return pred

    def predict_from_file(self, test_image_dir, 
                          test_df_file,
                          target_names,
                          batch_size,
                          size, 
                          **kwargs):
        pred = self._test_df.loc[self._index].tolist()
        self._index += 1
        return pred

    def save_model(self, **kwargs):
        pass
            
    def load_model(self, test_df_file, target_names, **kwargs):
        pred_names = [name.replace('pose','pred') for name in target_names]
        self._test_df = pd.read_csv(test_df_file)[pred_names]
        pass
    
    def print_model_summary(self):
        pass
    

def main():
    pass
    
if __name__ == '__main__':
    main() 