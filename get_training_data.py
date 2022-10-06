import glob
import os
import sys
class GetTrainingData:
    def __init__(self,path):
        self.training_data_list = []
        self.file_path = glob.glob(os.path.join(path,'*','*.dgp'))
        for i in self.file_path:
            data = i.split(os.sep)
            label = data[-2]
            image_pixel = []
            
            with open(i,'r',encoding='utf-8') as f:
                file = f.readlines()
                image_pixel = [0.0 if y == '0' else 1.0 for j in file for y in j if y == '0' or y == '1' ]
                self.training_data_list.append([image_pixel,label])
        
    def __getitem__(self,index1):
        return self.training_data_list[index1]
    
    def __len__(self):
        return len(self.training_data_list)
         
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index < len(self.training_data_list)-1:
            self.index += 1
            return self.training_data_list[self.index]
        else:
            raise StopIteration

if __name__ == '__main__':
    get_data = GetTrainingData('.\python\get_training_data\TrainData')
    
    for data in get_data:
        print(data)
        
            
            