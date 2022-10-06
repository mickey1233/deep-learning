import os
import sys
import glob

class MyData:
    def __init__(self,folder):
        self.data_list = []

        file_list = glob.glob(os.path.join(folder,'*','*.dgp'))
        for digit_path in file_list:
            all_tokens = []
            path_tokens = digit_path.split(os.sep)
            label = path_tokens[-2]
            with open(digit_path) as rfd:
                lines = rfd.readlines()
                for line in lines:
                    tokens = line.split()
                    all_tokens = all_tokens + tokens
            all_binary_tokens = [1.0 if xx != '0' else 0.0 for xx in all_tokens]
            self.data_list.append([all_binary_tokens,int(label)])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __iter__(self):
        self.inter_idx = 0
        return self

    def __next__(self):
        if self.inter_idx < len(self.data_list):
            idx = self.inter_idx
            self.inter_idx += 1
            return self.data_list[idx]
        else:
            raise StopIteration

if __name__ == '__main__':
    my_data = MyData('TrainData')

    for i, (data,label) in enumerate(my_data):
        print(i,len(data),data,label)


