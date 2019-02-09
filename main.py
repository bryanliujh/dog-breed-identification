from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.optimizers import Adam

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

RANDOM_SEED = 100
top_x_class = 120
EPOCHS_NUM = 10

df_train = pd.read_csv('labels.csv')
df_test = pd.read_csv('sample_submission.csv')

'''
# Take a look at the class/breed distribution
ax=pd.value_counts(df_train['breed'],ascending=True).plot(kind='barh',
                                                       fontsize="5",
                                                       title="Class Distribution",
                                                       figsize=(50,100))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()
'''

# Get the top 20 breeds for now so that it is less computation
top_breeds = sorted(list(df_train['breed'].value_counts().head(top_x_class).index))
df_train = df_train[df_train['breed'].isin(top_breeds)]

one_hot = pd.get_dummies(df_train['breed'], sparse=True)
one_hot_labels = np.asarray(one_hot)

# add the path name of the pics to the data set
df_train['image_path'] = df_train.apply( lambda x: ('train/' + x["id"] + ".jpg" ), axis=1)
print(df_train.head())

# Convert the images to arrays which is used for the model. Inception uses image sizes of 299 x 299
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in tqdm(df_train['image_path'].values.tolist())]).astype('float32')

# Split the data into train and test. The stratify parm will ensure train and test will have the same proportions of class labels as the input dataset
x_train, x_test, y_train, y_test = train_test_split(train_data, df_train['breed'], test_size=0.2, stratify=np.array(df_train['breed']), random_state=RANDOM_SEED)

print ('x_train shape = ', x_train.shape)
print ('x_test shape = ', x_test.shape)

'''
# Calculate the value counts for train and test data and plot to show a good stratify, the plot should show an equal percentage split for each class
data = y_train.value_counts().sort_index().to_frame()   # this creates the data frame with train numbers
data.columns = ['train']   # give the column a name
data['test'] = y_test.value_counts().sort_index().to_frame()   # add the test numbers
new_plot = data[['train','test']].sort_values(['train']+['test'], ascending=False)   # sort the data
new_plot.plot(kind='bar', stacked=True)
plt.show()

'''

# Need to convert the prediction train and test labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_test = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()




# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   # zoom_range = 0.3,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=10)


# Create test generator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = train_datagen.flow(x_test, y_test, shuffle=False, batch_size=10, seed=10)


# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))


# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with 20 classes
#(there will be 120 classes for the final submission)
x = Dense(512, activation='relu')(x)
predictions = Dense(top_x_class, activation='softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)

# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile with Adam
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = 175,
                      validation_data = test_generator,
                      validation_steps = 44,
                      epochs = EPOCHS_NUM,
                      verbose = 2)


# Use the sample submission file to set up the test data - x_test
# Create the x_test
x_test = []
for i in tqdm(df_test['id'].values):
    img = cv2.imread('test/{}.jpg'.format(i))
    x_test.append(cv2.resize(img, (299, 299)))

# Make it an array
x_test = np.array(x_test, np.float32) / 255.

print('xtest array')

# Predict x_test
predictions = model.predict(x_test, verbose=2)

print('prediction')

# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values

print('onehotdone')

# Create the submission data.
submission_results = pd.DataFrame(predictions, columns = col_names)

print('creating submission')

# Add the id as the first column
submission_results.insert(0, 'id', df_test['id'])

# Save the submission
submission_results.to_csv('submission.csv', index=False)


'''
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
im_size = 90
x_train = []
y_train = []
x_test = []
i = 0
for f, breed in tqdm(df_train.values):
    img = cv2.imread('train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1
for f in tqdm(df_test['id'].values):
    img = cv2.imread('test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))
y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.
print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)
num_class = y_train_raw.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)
# Create the base pre-trained model
base_model = VGG19(#weights='imagenet',
    weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)
# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()
model.fit(X_train, Y_train, epochs=1, validation_data=(X_valid, Y_valid), verbose=1)
preds = model.predict(x_test, verbose=1)
sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', df_test['id'])
sub.head(5)
'''