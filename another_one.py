from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

# data = pd.read_csv('my_data.csv')
# data = data[['text', 'sentiment']]

max_features = 2000
# tokenizer = Tokenizer(num_words=max_features, split=' ')
# tokenizer.fit_on_texts(data['text'].values)
# x = tokenizer.texts_to_sequences(data['text'].values)
# x = pad_sequences(X)
embed_dim = 128
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train)
x_test = pad_sequences(x_test)

model = Sequential()
model.add(Embedding(max_features, embed_dim))
# model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(200, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Y = pd.get_dummies(data['sentiment'].values)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

# batch_size = 10
# X_train = X_train[:3000]
# Y_train = Y_train[:3000]
# X_test = X_test[:3000]
# Y_test = Y_test[:3000]

batch_size = 32
model.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=2)
# validation_size = 1500

# y_validate = x_test[-validation_size:]
# x_validate = y_test[-validation_size:]
# x_test = x_test[:-validation_size]
# x_test = y_test[:-validation_size]

# scores = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
# print("Accuracy training: %.2f%%" % (scores[1] * 100))

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("Accuracy test: %.2f%%" % (scores[1] * 100))
