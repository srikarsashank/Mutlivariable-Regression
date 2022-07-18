import numpy as np
import csv
import pandas as pd
import random
from math import sqrt
from numpy import mean
from numpy.random import rand

N = 1650
train_N=1155
test_N=495
LOOPS = 1000
learning_rate = 0.8
degree = 2
lasso_lambda=1e-5
file=open('FoDS-A1.csv')
csvReader=csv.reader(file)
row=[]
row=next(csvReader)
strength_arr=[]
temperature_arr=[]
pressure_arr=[]


for itr in range(N):
    row=next(csvReader)
    strength_arr.append((float)(row[0]))
    temperature_arr.append((float)(row[1]))
    pressure_arr.append((float)(row[2]))

strength_max=max(strength_arr)
temperature_max=max(temperature_arr)
pressure_max=max(pressure_arr)
strength_min=min(strength_arr)
temperature_min=min(temperature_arr)
pressure_min=min(pressure_arr)


for itr in range(N):
    strength_arr[itr]=(strength_arr[itr]-strength_min)/(strength_max-strength_min)
    temperature_arr[itr]=(temperature_arr[itr]-temperature_min)/(temperature_max-temperature_min)
    pressure_arr[itr]=(pressure_arr[itr]-pressure_min)/(pressure_max-pressure_min)

df = pd.DataFrame(data={ "Strength" :strength_arr, "Temperature":temperature_arr,"Pressure":pressure_arr})
df.to_csv("output.csv", sep=',',index=False,header=False)
df.to_csv("normalized.csv", sep=',',index=False)

fid=open("output.csv","r")
li=fid.readlines()
fid.close()
random.shuffle(li)
fid=open("output2.csv","w")
fid.writelines(li)
fid.close()

strength_arr=[]
temperature_arr=[]
pressure_arr=[]

file=open('output2.csv')
csvReader=csv.reader(file)
row=[]

for itr in range(N):
    row=next(csvReader)
    strength_arr.append((float)(row[0]))
    temperature_arr.append((float)(row[1]))
    pressure_arr.append((float)(row[2]))

strength_arr_train=strength_arr[0:train_N]
strength_arr_test=strength_arr[train_N:]
temperature_arr_train=temperature_arr[0:train_N]
temperature_arr_test=temperature_arr[train_N:]
pressure_arr_train=pressure_arr[0:train_N]
pressure_arr_test=pressure_arr[train_N:]

# print(pressure_arr_test)





weight = np.ones(shape=(degree+1, degree+1))
for i in range(0,degree+1):
    for j in range(0,degree+1):
        weight[i][j]=(float)(random.uniform(-(1.0 / sqrt(train_N)),(1.0 / sqrt(train_N))))
print(weight)

for epoch in range(LOOPS):

    # Regresion Step
    error_de=0
    error_test=0
    # Regresion Step
    add=0
    pressure_pred  = np.zeros(shape=(train_N,1)) #from regression
    if(epoch%50==0):
        for i in range(degree+1):
            for j in range(degree+1):
                    if(i + j <= degree):
                        add+=(weight[i][j])**2
    add*=lasso_lambda
    add/=2
        
    for k in range(train_N):
        strength=strength_arr_train[k]
        temperature=temperature_arr_train[k]
        # print("s =" + (str)(strength))
        # print("t =" + (str)(temperature))
        for i in range(degree+1):
            for j in range(degree+1):
                    if(i + j <= degree):
                        pressure_pred[k] += weight[i][j] * (strength ** i) * (temperature ** j)
        pressure_final=pressure_pred[k]*(pressure_max-pressure_min)+pressure_min
        pressure_denorm=pressure_arr_train[k]*(pressure_max-pressure_min)+pressure_min
        error_de+=(pressure_final- pressure_denorm)**2
    error_de+=add
    error_de/=train_N
    error_de/=2
    pressure_pred1  = np.zeros(shape=(train_N,1))
        
    for k in range(test_N):
        strength1=strength_arr_test[k]
        temperature1=temperature_arr_test[k]
        for i in range(degree+1):
                for j in range(degree+1):
                        if(i + j <= degree):
                            pressure_pred1[k] += weight[i][j] * (strength1 ** i) * (temperature1 ** j)
        pressure_final=pressure_pred1[k]*(pressure_max-pressure_min)+pressure_min
        pressure_denorm=pressure_arr_test[k]*(pressure_max-pressure_min)+pressure_min
        error_test+=(pressure_final- pressure_denorm)**2
    error_test+=add
    error_test/=test_N
    error_test/=2
    if(epoch%50==0):
        print("training error =" +(str)(error_de))
        print("testing error = "+ (str)(error_test))
        
    # # Calculating the ERROR
    # if(epoch%50==0 and epoch!=0):
    #     error = 0
    #     for i in range(train_N):
    #         error += (abs(pressure_pred[i]-pressure_arr_train[i]))
    #     # error = error /(2*1155)
    #     error/=train_N
    #     # print(error)
    #     # print("error =" + (str)(error))
    # # LEARNING STEP
    for i in range(degree+1):
        for j in range(degree+1):
            if(i + j <= degree):
                temp = 0
                for k in range(train_N):
                    temp = temp+((pressure_pred[k] - pressure_arr_train[k])* (strength_arr_train[k] ** i) * (temperature_arr_train[k] ** j))
                if weight[i][j]>0:
                    temp+=lasso_lambda
                else :
                    temp-=lasso_lambda
                weight[i][j] = weight[i][j] - (((learning_rate) * (temp))/train_N)
  

print(weight)



temp_final=temperature_arr_train+temperature_arr_test
strength_final=strength_arr_train+strength_arr_test
pressure_final=pressure_arr_train+pressure_arr_test

for i in range(N):
    temp_final[i]=temp_final[i]*(temperature_max-temperature_min)+ temperature_min
    pressure_final[i]=pressure_final[i]*(pressure_max-pressure_min)+ pressure_min
    strength_final[i]=strength_final[i]*(strength_max-strength_min)+ strength_min
df = pd.DataFrame(data={ "Strength" :strength_final, "Temperature":temp_final,"Pressure":pressure_final})
df.to_csv("final_output.csv", sep=',',index=False,header=True)


    




















