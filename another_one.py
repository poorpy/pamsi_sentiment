import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

data = pd.read_csv('my_data.csv')
data = data[['text', 'sentiment']]

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(84, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

batch_size = 10
X_train = X_train[:5]
Y_train = Y_train[:5]
X_test = X_test[:5]
Y_test = Y_test[:5]
# batch_size = 32
model.fit(X_train, Y_train, epochs=3, batch_size=batch_size, verbose=2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
# score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))
# print("Accuracy: %.2f%%" % (scores[1]*100))

scores = model.evaluate(X_train, Y_train, batch_size=batch_size, verbose=0)
print("Accuracy training: %.2f%%" % (scores[1] * 100))

scores = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print("Accuracy test: %.2f%%" % (scores[1] * 100))
