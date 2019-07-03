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
num_pixels = X_train.shape[1]*X_train.shape[2]
num_pixels = 9
#plt.imshow(X_pred[0])
#plt.imshow(X_pred[1])
#plt.show()

# Processando data de entrada
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)
X_predrs = X_pred.reshape(X_pred.shape[0], num_pixels)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_predrs = X_predrs.astype('float32')
X_train /= 255
X_test /= 255
X_predrs /= 255

#a = input()
# Processando classes e modelos
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

num_classes = Y_test.shape[1]
second_layer = int(num_pixels/4)
third_layer = int(num_pixels/8)

# Definindo a arquitetura
model = Sequential()
model.add(Dense( 9, input_dim = 9, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense( second_layer, input_dim = 9, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense( third_layer, input_dim = second_layer, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense( num_classes, kernel_initializer = 'normal', activation='softmax', name = 'preds'))

# Compilando o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo com os dados de treino
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size=100, verbose=1)

# Treinando o modelo com os dados de teste
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
pred = model.predict_classes(X_predrs, verbose = 1)
idx = range(len(X_pred))
for i in idx:
	a = input('ENTER p/ reconhecer um digito.')
	plt.imshow(X_pred[i])
	plt.show()
	print(pred[i])
