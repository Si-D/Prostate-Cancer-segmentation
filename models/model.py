import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Flatten

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

SIZE_X = 128 
SIZE_Y = 128
n_classes=5

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("augmented_more_images/images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = []
for directory_path in glob.glob("augmented_more_images/masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_masks = np.array(train_masks)
#print(train_images.shape, train_masks.shape)


Green = np.array([0,255,0])
Blue = np.array([0,0,255])
Yellow = np.array([255,255,0])
Red = np.array([255,0,0])
Unlabelled = np.array([255,255,255])

j = 0
while j < (train_masks.shape[0]):
    label = train_masks[j]
    def rgb_to_2D_label(label):
        """
        Supply our labale masks as input in RGB format. 
        Replace pixels with specific RGB values ...
        """
        label_seg = np.zeros(label.shape,dtype=np.uint8)
        label_seg [np.all(label == Green,axis=-1)] = 0
        label_seg [np.all(label==Blue,axis=-1)] = 1
        label_seg [np.all(label==Yellow,axis=-1)] = 2
        label_seg [np.all(label==Red,axis=-1)] = 3
        label_seg [np.all(label==Unlabelled,axis=-1)] = 4
        
        label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
        
        return label_seg
    j +=1
#print(label_seg.shape)
    
labels = []
for i in range(train_masks.shape[0]):
    label = rgb_to_2D_label(train_masks[i])
    labels.append(label) 
    i +=1
        
    
labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)
 

print("Unique labels in label dataset are: ", np.unique(labels))


n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)

train_images.sort()
train_masks.sort()
# labelencoder = LabelEncoder()
# n, h, w = train_masks.shape
# train_masks_reshaped = train_masks.reshape(-1,1)
# train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
# train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

# np.unique(train_masks_encoded_original_shape)
 

#################################################
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)

train_masks_input = labels_cat
# print(X[1].shape, Y.shape)
# print(Y[1].shape)

X = train_images
Y = train_masks_input


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

# =============================================================================
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(X_train[image_number], (128,128,-1)))
# plt.subplot(122)
# plt.imshow(np.reshape(Y_train[image_number], (128,128,-1)))
# plt.show()
# =============================================================================


# train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
# y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))

# test_masks_cat = to_categorical(Y_test, num_classes=n_classes)
# y_test_cat = test_masks_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], n_classes))

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.20, 0.20, 0.20, 0.20, 0.20]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

smooth=1

def dice_coef(y_true, y_pred):
    y_true_f = Flatten(y_true)
    y_pred_f = Flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
#optimizer = Adam(lr=0.00001)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss=dice_loss, metrics=[metrics])
print(model.summary())

early_stop = EarlyStopping(monitor='val_iou_score', patience=20, verbose=1)
callbacks_list = [early_stop]

history=model.fit(X_train,
                Y_train,
                batch_size=8,
                epochs=150,
                verbose=1,
                validation_data=(X_val, Y_val),
                )

model.save('test.hdf5')


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

X_test1 = X_test[:2000]

_,val, acc = model.evaluate(X_test1, Y_val)
print("Accuracy is = ", (acc * 100.0), "%")

model.load_weights('test.hdf5')

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())




# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=Y_test[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img_norm, 0)
# prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

# test_img_other = cv2.imread('data/test_images/02-1_256.tif', 0)
# #test_img_other = cv2.imread('data/test_images/img8.tif', 0)
# test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
# test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
# test_img_other_input=np.expand_dims(test_img_other_norm, 0)

# #Predict and threshold for values above 0.5 probability
# #Change the probability threshold to low value (e.g. 0.05) for watershed demo.
# prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')
# plt.subplot(234)
# plt.title('External Image')
# plt.imshow(test_img_other, cmap='gray')
# plt.subplot(235)
# plt.title('Prediction of external Image')
# plt.imshow(prediction_other, cmap='gray')
# plt.show()


test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=Y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input(test_img_input)

test_pred1 = model.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,4])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1)
plt.show()




