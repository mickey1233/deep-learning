from save_load_model import save_load_model
import pathlib
import os
import MyData 
import MyLoss
import time
from training_model import MLP
if __name__ == "__main__":
    ##GET DATA##
    folder_path = pathlib.Path(__file__).parent.absolute()
    path = os.path.join(folder_path,'TrainData')
    #data = gtd.GetTrainingData(path)
    data = MyData.MyData(path)
    
    ##LOAD MODEL##
    mlp = MLP(64,10)
    loss = MyLoss.CrossEntropyLoss()
    folder_path = pathlib.Path(__file__).parent.absolute()
    load_model = save_load_model()
    a = os.path.join(folder_path,'train_model')
    #print(a)
    load_model.load_model(os.path.join(folder_path,'train_model'),mlp)
    count = 1
    acc = 0
    start_time = time.time()
    for (train_data,label) in data:
        start1_time = time.time()
        train_list = []
        
        actual_data = [1.0 if i == label else 0.0 for i in range(10)]
        train_model =  mlp(train_data)
        train_loss = loss(actual_data,train_model)
        for i in train_model:
            train_list.append(i.value)
        end1_time = time.time()
        print('第{}筆data'.format(count),'預測結果 =',train_list,'最大值 =',max(train_list),'分類結果 = ',train_list.index(max(train_list)),'實際分類 =',label,'loss =',train_loss.value,'預測花費時間 =', end1_time-start1_time)
        if train_list.index(max(train_list)) == label:
            acc+=1
        count+=1
    end_time = time.time()
    count-=1
    print('預測正確數量 = ', acc,'總數量 =', count,'準確度 = ', acc/count)
    print('預測總花費時間 =', end_time-start_time)