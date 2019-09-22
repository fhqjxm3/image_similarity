# image_similarity

This is an example of finding similar images using deep-leaning models.

# How ?

Here i used a simple classifier and used it's last convolution and flattend.
Which will give us a vector and i used cosine similarity to check whether
the vectors produced by our model are equal or not.There are two models.
One is a custom classifer made of custom architecture and trained and tested on
fashion-mnist dataset the other one is a pretrained vgg on imagenet which gives 
us a lot of classes for checking the similarity out of the box.In vgg i have created
a small flask server which will gives us the similarity model as an api

# How to run ?

### For try out the vgg model just run <br/>
`python3 pretrained_model.py` <br/>
Which will give a server running at port 3000 the request format is shown below <br/>
![](imgs/postman.png)
<br/>
### For training  the fashion mnist model <br/>
`python3 trainer.py --num_epochs=100` or `python3 trainer.py`<br/>
The default epoch is set to 10 <br/>
### For testing it with the already provided model <br/>
`python3 infer.py`
