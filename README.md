# CarND1 Project3
The required five files for the project are included here.

After an inordinate amount of time trying to "correctly" train my neural network I decided to back off a bit and heed the advice I've received before: K.I.S.S (Keep It Simple, Stupid!). I had built up a network similar in complexity to the Nvidia one (cited in the End to End Learning for Self-Driving Cars), that did not drive the car even on Track 1. My model parameters file was about 180MB, and it could'nt drive!
So, I scaled back to just 1 Convolution Layer with MaxPooling and one Fully Connected output layer; model parameter file was now just 1.1MB, and it drove around most of Track 1. Finally, some progress!!
The model as submitted drives around Track 1 at 30mph.

I am working on augmenting the dataset with data from Track2, to help generalize more; but I decided to proceed to submit the project, right now as it should meet all the requirements in the rubric.
