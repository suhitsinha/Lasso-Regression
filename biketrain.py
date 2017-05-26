import pandas as pd
import numpy as np

from sklearn.svm import SVR

data1 = pd.read_csv('bikeDataTrainingUpload.csv',header=0)
data2 = pd.read_csv('bikeDataTrainingUpload.csv',header=0)

data1['temp'] = np.power(data1['temp'], 3);
data1['atemp'] = np.power(data1['atemp'], 1/3);
data1['hum'] =  np.power(data1['hum'], 1/2);
data1['windspeed'] = np.power(data1['windspeed'], 2);

data2['temp'] = np.power(data2['temp'], 3);
data2['atemp'] = np.power(data2['atemp'], 1/3);
data2['hum'] =  np.power(data2['hum'], 1/2);
data2['windspeed'] = np.power(data2['windspeed'], 2);

dumData1 = []
dumData2 = []
nomCols = ['season','yr','holiday','mnth','weekday','workingday','weathersit']

for col in nomCols:
	dumData1.append(pd.get_dummies(data1[col]))

for col in nomCols:
	dumData2.append(pd.get_dummies(data2[col]))

newDumData1 = pd.concat(dumData1, axis=1)
newDumData2 = pd.concat(dumData2, axis=1)

data1 = pd.concat((data1,newDumData1),axis=1)
data2 = pd.concat((data2,newDumData2),axis=1)


data1 = data1.drop(['season','yr','mnth','holiday','weekday','workingday','weathersit'],axis=1)
data2 = data2.drop(['season','yr','mnth','holiday','weekday','workingday','weathersit'],axis=1)

X1 = data1.values
X2 = data2.values
Y1 = data1['casual'].values
Y2 = data1['registered'].values

eliminate = [4, 5, 6]
X1 = np.delete(X1,eliminate,axis=1)
X2 = np.delete(X2,eliminate,axis=1)

#X_train1, X_test1, Y_train1, Y_test1 = cross_validation.train_test_split(X1,Y1,test_size=0.01,random_state=0)
#X_train2, X_test2, Y_train2, Y_test2 = cross_validation.train_test_split(X2,Y2,test_size=0.01,random_state=0)

regLassoCasY1 = SVR(kernel='rbf', C=549000, gamma=.006)
regLassoCasY2 = SVR(kernel='rbf', C=549000, gamma=.006)

regLassoCasY1.fit(X1, Y1)
regLassoCasY2.fit(X2, Y2)

#------------------------------------------PREDICTION---------------------------------------------------#

predictData1 = pd.read_csv('TestX.csv',header=0)
predictData2 = pd.read_csv('TestX.csv',header=0)

preDumData1 = []
preDumData2 = []

for col in nomCols:
	preDumData1.append(pd.get_dummies(predictData1[col]))

for col in nomCols:
	preDumData2.append(pd.get_dummies(predictData2[col]))

newpreDumData1 = pd.concat(preDumData1, axis=1)
newpreDumData2 = pd.concat(preDumData2, axis=1)

predictDataX1 = pd.concat((predictData1,newpreDumData1),axis=1)
predictDataX2 = pd.concat((predictData2,newpreDumData2),axis=1)

predictDataX1['temp'] = np.power(predictDataX1['temp'], 3);
predictDataX1['atemp'] = np.power(predictDataX1['atemp'], 1/3);
predictDataX1['hum'] = np.power(predictDataX1['hum'], 1/2);
predictDataX1['windspeed'] = np.power(predictDataX1['windspeed'], 2);

predictDataX2['temp'] = np.power(predictDataX2['temp'], 3);
predictDataX2['atemp'] = np.power(predictDataX2['atemp'], 1/3);
predictDataX2['hum'] = np.power(predictDataX2['hum'], 1/2);
predictDataX2['windspeed'] = np.power(predictDataX2['windspeed'], 2);

predictDataX1 = predictDataX1.drop(['season','yr','mnth','holiday','weekday','workingday','weathersit'],axis=1)
predictDataX2 = predictDataX2.drop(['season','yr','mnth','holiday','weekday','workingday','weathersit'],axis=1)

#-------------------------------Predict and Fit-------------------------------------------------------------

predictY1 = regLassoCasY1.predict(predictDataX1)

predictY2 = regLassoCasY2.predict(predictDataX2)

predictCnt = predictY1 + predictY2

predRound = np.around(predictCnt, decimals=0)

lenParray = len(predRound)

for x in np.nditer(predRound, op_flags=['readwrite']):
	if x[...]<0:
		x[...] = 0

seqno = []
seqno.append(('ID', 'CNT'))

for i in range(lenParray):
	seqno.append((int(i), int(predRound[i])))

np.savetxt("153050005.csv", seqno, delimiter=",", fmt='%s')


