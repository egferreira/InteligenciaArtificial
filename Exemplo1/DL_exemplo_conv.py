# Importanto Libs 
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
 
# Carregando pre-embaralhado MNIST data como treino e teste
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_pred = X_train[50000:59000, 0:28, 0:28]
X_train = X_train[0:49000, 0:28, 0:28]
Y_train = Y_train[0:49000]
#plt.imshow(X_pred[0])
#plt.imshow(X_pred[1])
#plt.show()

# Processando data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_predrs = X_pred.reshape(X_pred.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_predrs = X_predrs.astype('float32')
X_train /= 255
X_test /= 255
X_predrs /= 255

#a = input()

# Classificando datas
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

 
# Definindo modelo de arquitetura
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), dim_ordering='th'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# Compilando o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Trainando  modelo com os dados de treino
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)
 
# Testando o modelo com os dados de teste
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
pred = model.predict_classes(X_predrs, verbose = 1)
idx = range(len(X_pred))
for i in idx:
	a = input('ENTER p/ reconhecer um digito:')
	plt.imshow(X_pred[i])
	plt.show()
	print(pred[i])