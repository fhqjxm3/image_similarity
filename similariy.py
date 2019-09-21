from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity as cosine
import math
import numpy as np
from PIL import Image
import sys


def predict_similarity(image1, image2):
	'''
	This function calculates the similarity of two images
	the model used is pretrained vgg on imagenet dataset
	we eliminate the last layers and take the output from
	last convolution layer then flattens it to calculate 
	similarity
	'''
	pretrained_model = VGG16(include_top=False,weights="imagenet", input_shape=(224, 224, 3))
	data1 = np.array(Image.open(image1).convert('RGB').resize((224, 224)))
	data2 = np.array(Image.open(image2).convert('RGB').resize((224, 224)))
	data1 = (data1.reshape(-1, 224, 224, 3) / 255).astype(np.float32)
	data2 = (data2.reshape(-1, 224, 224, 3) / 255).astype(np.float32)
	pred1 = pretrained_model.predict(data1).reshape(1,-1)
	pred2 = pretrained_model.predict(data2).reshape(1,-1)
	sim = cosine(pred1,pred2)
	print("similarity between images {} ".format(round(sim.item(0),2)))
	return round(sim.item(0),2)
	# print(pretrained_model)


if __name__ == "__main__":
	img1 = sys.argv[1]
	img2 = sys.argv[2]
	# print(type(img1),type(img2))
	predict_similarity(img1, img2)