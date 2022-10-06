import get_training_data as gtd
import loss_caculation as lc
import training_model as tm
import MyOptim
import sys
import matplotlib.pyplot as plt


if __name__ == "__main__":
    total_loss_list = []
    epoch_list = []
    data = gtd.GetTrainingData('D:\work\python\homework\TrainData')
    #data = gtd.GetTrainningData('http://127.0.0.1:5000')
    model = tm.MLP(len(data),10) #0~9
    loss = lc.Loss()
    optim = MyOptim.MyOptim(model.parameters())
    #在每個epoch計算loss
    for epoch in range(10): 
        total_loss = 0
        
        for (train_data,label) in data:
            model.zero_grad()
            actual_data = [1.0 if i == label else 0.0 for i in range(10)] #0~9
            #print(train_data,label)
            train_model =  model(train_data)
            #print(train_model)
            train_loss = loss(actual_data,train_model)
            train_loss.backward()
            optim.step()
            total_loss += train_loss.value
        print(epoch+1,total_loss)
        total_loss_list.append(total_loss)
        epoch_list.append(epoch)
    plt.plot(epoch_list,[0,1])
    plt.title("loss",fontsize=24)
    plt.xlabel("epoch",fontsize=14)
    plt.ylabel("total_loss",fontsize=14)
    plt.show()