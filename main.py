from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.optimizers import Adam

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

#The goal of this kaggle competition is to minimise log loss between the predicted probability and observed target
#The submission will be a predictions of probability each dog will belong to a certain breed (e.g sample_submission.csv)
#After submission kaggle will do the calculation of the log loss automatically
#There are a total of 120 breeds of dogs

#what parameters to choose? https://www.kaggle.com/c/dog-breed-identification/discussion/46201
#https://www.kaggle.com/c/dog-breed-identification/discussion/43203


#seed value initialise the randomnisation, you can set any values for the seed
# using the same seed number guarantees the CNN will produce the same result all the time

#https://stackoverflow.com/questions/45943675/meaning-of-validation-steps-in-keras-sequential-fit-generator-parameter-list
#steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
#validation_steps = TotalvalidationSamples / ValidationBatchSize

BREED_NUM = 120
EPOCHS_NUM = 8



#Dimensions (height x weight) to be no smaller than 75
DIMENSIONS = 299

#https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
BATCH_SIZE = 35

df_train = pd.read_csv('labels.csv')
df_test = pd.read_csv('sample_submission.csv')


# You define the top X breeds by changing BREED_NUM so that it is less computation during testing, but for submission must use 120
top_breeds = sorted(list(df_train['breed'].value_counts().head(BREED_NUM).index))
df_train = df_train[df_train['breed'].isin(top_breeds)]

one_hot = pd.get_dummies(df_train['breed'], sparse=True)
one_hot_labels = np.asarray(one_hot)

# add the path name of the pics to the data set
df_train['image_path'] = df_train.apply( lambda x: ('train/' + x["id"] + ".jpg" ), axis=1)
print(df_train.head())

# Convert the images to arrays which is used for the model. Inception uses image sizes of 299 x 299
train_data = np.array([img_to_array(load_img(img, target_size=(DIMENSIONS, DIMENSIONS))) for img in tqdm(df_train['image_path'].values.tolist())]).astype('float32')

# Split the data into train and test. The stratify parm will ensure train and test will have the same proportions of class labels as the input dataset
x_train, x_validation, y_train, y_validation = train_test_split(train_data, df_train['breed'], test_size=0.2, stratify=np.array(df_train['breed']), random_state=100)

print ('x_train shape = ', x_train.shape)
print ('x_validation shape = ', x_validation.shape)

train_sample_size, n, p, q = x_train.shape
validation_sample_size, n, p, q = x_validation.shape

VALIDATION_STEPS = validation_sample_size // BATCH_SIZE
STEPS_PER_EPOCH = train_sample_size // BATCH_SIZE

print(VALIDATION_STEPS)

# Need to convert the prediction train and test labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()



#image generator is used to handle the image manipulation eg. shearing, flipping
# https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad
# Create train generator.
#set shuffle to False, because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=BATCH_SIZE, seed=10)


# Create validation generator
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=BATCH_SIZE, seed=10)

#https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(DIMENSIONS, DIMENSIONS, 3))
#base_model = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape=(DIMENSIONS, DIMENSIONS, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with X breeds (classes)
#(there will  120 breeds (classes) for the final submission)
x = Dense(512, activation='relu')(x)
predictions = Dense(BREED_NUM, activation='softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)

# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

#possible to experiment with SGD?
# Compile with Adam optimisation, learning rate = 0.0001
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = STEPS_PER_EPOCH,
                      validation_data = validation_generator,
                      validation_steps = VALIDATION_STEPS,
                      epochs = EPOCHS_NUM,
                      verbose = 2)


# Use the sample submission file to set up the test data - x_test
# Create the x_test
x_test = []
for i in tqdm(df_test['id'].values):
    img = cv2.imread('test/{}.jpg'.format(i))
    x_test.append(cv2.resize(img, (DIMENSIONS, DIMENSIONS)))

# Make it an array
x_test = np.array(x_test, np.float32) / 255.

print('Please wait for predictions to be done..........')

# Predict x_test
predictions = model.predict(x_test, verbose=2)


# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values


# Create the submission data.
submission_results = pd.DataFrame(predictions, columns = col_names)


# Add the id as the first column
submission_results.insert(0, 'id', df_test['id'])

# Save the submission
submission_results.to_csv('submission.csv', index=False)


print('Submission Created')