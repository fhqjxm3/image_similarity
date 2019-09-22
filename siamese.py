import keras.datasets.fashion_mnist as fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from keras.models import Sequential,Model
import numpy as np
# load dataset

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# reshape the data for the network
x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype('float32')
x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype('float32')


train_groups = [x_train[np.where(y_train==i)[0]]for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)]
[print(x.shape) for x in train_groups]


def random_batch_mix(group, batch_sizehalf = 8):
	'''
	A function to make our dataset suited for us 
	Basically the dataset that splitted into groups
	are concatibated together with another image in
	same of different group if it is form same group
	the the similarity value will be 1 else 0

	inputs
	======
	group  : grouped images in test or train
	batch_sizehalf : batch size basically each
	for a true and flase case will be generated
	so the batch_sizehalf

	output
	======
	The concatinated image and its similarities
	'''
	img_a, img_b, score = [], [], []
	total_class = list(range(len(group)))
	for match in [True,False]:
		# print(batch_sizehalf)
		group_idx = np.random.choice(total_class, size=int(batch_sizehalf))
		img_a += [group[item_index][np.random.choice(range(group[item_index].shape[0]))]for item_index in group_idx]
		if match:
			b_group_index = group_idx
			score += [1]*batch_sizehalf
		else:
			b_group_index = [np.random.choice([i for i in total_class if i!= item_index])for item_index in group_idx]
			score +=[0]*batch_sizehalf
		img_b += [group[item_index][np.random.choice(range(group[item_index].shape[0]))]for item_index in b_group_index]
	return np.stack(img_a, 0), np.stack(img_b, 0), np.stack(score, 0)



def feature_extractor():
	'''
	The feature extractor model whose prediction ouputs will
	be used as the input for the similarity model
	'''
	input_shape = x_train.shape[1:]
	model = Sequential()
	model.add(Conv2D(
			filters=32, kernel_size=(5, 5), activation="relu",
			input_shape=input_shape))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.3))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.6))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	return model

feature_model = feature_extractor()
feature_model.summary()
# print(x_train.shape[1:])
# Siamese model

def data_siam(group, batch_size=32):
	'''
	Dataset generator for similarity model
	'''
	while True:
		batch_img_a, batch_img_b, batch_sim = random_batch_mix(train_groups, batch_size//2)
		yield [batch_img_a, batch_img_b], batch_sim


def similarity_model():
	'''
	The similarity model use the extracted features to 
	learn the similariy
	'''
	img_a_in = Input(shape = x_train.shape[1:])
	# print(img_a_in)
	img_b_in = Input(shape = x_train.shape[1:])
	predicted_a = feature_model(img_a_in)
	predicted_b = feature_model(img_b_in)
	extracted = concatenate([predicted_a, predicted_b])
	extracted = Dense(16, activation = 'linear')(extracted)
	extracted = BatchNormalization()(extracted)
	extracted = Activation('relu')(extracted)
	extracted = Dense(4, activation = 'linear')(extracted)
	extracted = BatchNormalization()(extracted)
	extracted = Activation('relu')(extracted)
	extracted = Dense(1, activation = 'sigmoid')(extracted)
	siamese_model = Model(inputs = [img_a_in, img_b_in], outputs = [extracted], name = 'Similarity_Model')
	siamese_model.summary()
	# Siamese_model(predicted_a)
	siamese_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['mae'])

	valid_a, valid_b, valid_sim = random_batch_mix(test_groups, 512)
	loss_history = siamese_model.fit_generator(data_siam(train_groups),
								   steps_per_epoch = 700,
								   validation_data=([valid_a, valid_b], valid_sim),
												  epochs = 10,
												 verbose = True)


	print("Training completed")
	test_a, test_b, test_sim = random_batch_mix(test_groups, 4)
	prediction = siamese_model.predict([test_a,test_b])
	# print(prediction, test_sim)
	for i in range(len(test_sim)):
		print("actual similarity : {} predicted similarity : {}".format(test_sim[i],np.array(prediction).item(i)))
similarity_model()