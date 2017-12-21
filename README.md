# M042
I tried to split project into different parts and explain which file is from which part. There may be some manual work in between these functions that I forgot to mention, so feel free to ask me if something seems off. Also some of it still needs cleaning/commenting, so sorry in advance.
# 1- Reading SVS files and saving .png/.tif images of desired layer to be fed to CellProfiler
This should be readSVS and extractLayer
# 2- Global matching
This should be matchPhoto. There is also another version matchPhotoHessian, but I haven't tested that one out extensively so I am not going to upload it for now
# 3- Local matching
This should be XandYmatching. I tried my idea of comparing cells (feature vectors) from H&E images to cells from TP53 images, but it didn't end up working well so I haven't uploaded that version, this is the no local matching / assume global matching works perfectly version. I will update and upload a commented/cleaned out version later
# 4- Label generation
This should be EM_labelf. Of course one thing to remember is that the labels may be reversed, so if that is the case, we reverse labels before progressing to next steps
There is also a script called EM_vis that we use to visualize cells with their labels, we use it to see how well EM worked. Its final version is pretty messy because we were trying to get a good image to include in the report, I will upload it after cleaning it up and maybe turning it into a function as well
# 5- PCA
This is PCA_custom. Actually this is based on the MATLAB implementation of PCA, the only difference is that there is an option to apply variance normalization on the features before PCA
# 6- Training SVM and logistic regression using leave-one-out cross validation
This is hold1CV_pca. The code should generally be easy to follow, but the initial version is uncommented. I would recommend waiting for me to add comments if there are unclear parts
# 7- Training neural net
These are in python (tensorflow), neural_profiler1 and neural_train. I tried to make the code neater in this section because parts 1, 2, 3, 4 and 5 will be run once and then they will never get used again and 6 is more of a preliminary work than future direction. but we are probably going to tweak our neural networks a lot, add stuff in, remove stuff etc so I felt it was more important to have better code here. Hopefully as a result these will be easier to follow, and I will add comments probably over the weekend. The network is defined in neural_profiler1, and the way to train it is to call neural_train from terminal, but the final version of neural_train is actually for testing if I remember correctly (as I was trying to get results for the report). I will split neural_train into 2 different scripts, one for training, one for testing in the near future
