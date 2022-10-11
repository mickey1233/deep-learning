import pickle
from training_model import MLP
import pathlib
import os  
class save_load_model:
    def __init__(self):
        self.folder_path = pathlib.Path(__file__).parent.absolute()
    
    def save_model(self,modelname,parameters):
        parameters_dict = {}
        path = os.path.join(self.folder_path,modelname)
        for idx,parameter in enumerate(parameters):
            parameters_dict[idx] = parameter.value
        with open(path, 'wb') as f:
            #print(parameters_dict)
            pickle.dump(parameters_dict, f)
        
    def load_model(self,modelname,mlp):
        #parameters_dict = {}
        parameters_list = []
        path = os.path.join(self.folder_path,modelname)
        with open(path, 'rb') as f:
            parameters_dict = pickle.load(f)
            #print(parameters_dict)
            #try:
            #    parameters_dict = pickle.load(f)
            #except EOFError:
            #    return None   
        for parameters in parameters_dict.keys():
            parameters_list.append(parameters_dict[parameters])
        #print(parameters_list)
        mlp.parameters = parameters_list
        #print(mlp.parameters)



             
        
        
    