from keras.models import Model,load_model
import keras.datasets.fashion_mnist as fashion_mnist
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
 

def infer_images():
	'''
	This function uses the trained model and predict the similarity
	between two test samples
	'''
	trained_model = load_model('custom_model.h5')
	''' 
	created a keras model instance using our desired input and output
	Mainly because of usability
	'''
	similarity_model = Model(
			inputs=trained_model.input, 
			outputs=trained_model.get_layer(trained_model.layers[-3].name).output)

	'''
	loaded the dataset
	'''
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	'''
	Reshape inputs 
	'''
	one = similarity_model.predict(x_test[0].reshape((-1, 28, 28, 1)))
	two = similarity_model.predict(x_test[1].reshape((-1, 28, 28, 1)))
	print("Similarity between the above : {} ".format(round(cosine(one, two), 2)))
	return round(cosine(one, two), 2)

if __name__ == "__main__":
	infer_images()
