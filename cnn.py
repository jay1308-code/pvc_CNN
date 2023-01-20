import tensorflow as tf
from tensorflow.keras import layers

# Create a sequential model
model = tf.keras.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# Add a fully connected layer
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))

# Add an output layer
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])

# Create an ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Create a generator for the training set
train_generator = train_datagen.flow_from_directory(
        'PVC/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# Create an ImageDataGenerator for the validation set
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        'PVC/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=val_generator,
      validation_steps=50)

model.save("PVC.h5")
print("model saved!!!")