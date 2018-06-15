import numpy
from keras.datasets import imdb
from keras.layers import Dense, Flatten, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

seed = 7
numpy.random.seed(seed)
top_words = 5000
(X1_train, Y1_train), (X1_test, Y1_test) = imdb.load_data(num_words=top_words)

max_words = 500
X1_train = pad_sequences(X1_train, maxlen=max_words)
X1_test = pad_sequences(X1_test, maxlen=max_words)


print(X1_train.shape)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(SpatialDropout1D(0.4))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), epochs=2, verbose=2)

scores = model.evaluate(X1_test, Y1_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
