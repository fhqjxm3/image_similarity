# from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity as cosine
import math
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from similariy import predict_similarity
import matplotlib.pyplot as plt 

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
	return "Image Similarity API online"

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
	if request.files.get('image1') == None or request.files.get('image2') == None:
		return "No image/images found"
	else:
		img1 = plt.imread(request.files.get('image1'))
		img2 = plt.imread(request.files.get('image2'))
		# print(type(img1))
		plt.imsave('test1.jpg',img1)
		plt.imsave('test2.jpg', img2)
		sim  = predict_similarity('test1.jpg', 'test2.jpg')
		return "Similarity between the given images {} ".format(sim)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=3000, debug=True, threaded=False)