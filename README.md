# Image-Captioning
Generates descriptions(captions) for a given image(5 for each image) using keras library. Dataset used is Flickr8k.

Steps-
1. Extracting image features using pretrained InceptionV3 model and dumping them in the pickle file.
2. Preparing text data for captions by cleaning the descriptions data, tokenizing and creating vocabulary of english words.
3. Loading the text and image data prepared in the above 2 steps.
4. Defining the model.
5. Fitting the model on the training data and validating it on the validation data.
6. Generating the new descriptions for the test images using both normal max search and beam search(for beam_indexes 3,5,7).
7. Computing the BLEU score that summarizes how close the generated text is to the expected text.

To download the dataset- https://forms.illinois.edu/sec/1713398
