# WP1.2
Plot 2D sketch model of blended wing aircarft concept model and transform into 3D usinng Tensor flow algorithm


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input from the designer
x1 = np.array([20, 50, 70, 90, 110, 130, 150, 170, 190])
y1 = np.array([30, 80, 120, 150, 190, 220, 250, 280, 310])

# Create the TensorFlow graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Model parameters
W = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

# Linear model
y_predict = W*X + b

# Loss
loss = tf.reduce_mean(tf.square(y_predict - Y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()


# Run
with tf.Session() as sess:
    sess.run(init)
    
    # Train
    for epoch in range(1000):
        sess.run(train, {X: x1, Y: y1})
        
    # Get the optimized parameter
    final_w, final_b = sess.run([W, b])
    
# Plot the result
x1_test = np.linspace(20, 190, 10)
y1_test = final_w*x1_test + final_b

plt.plot(x1_test, y1_test, 'r-', label='2D Sketch Model')
plt.plot(x1, y1, 'bo', label='Designer Input')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Transform model into 3D
X3 = tf.placeholder(tf.float32)
Y3 = tf.placeholder(tf.float32)
Z3 = tf.placeholder(tf.float32

# Model parameters
Wz = tf.Variable(0.0, name='weight_z')
bz = tf.Variable(0.0, name='bias_z')

# Linear model
z_predict = Wz*X3 + bz

# Loss
loss3 = tf.reduce_mean(tf.square(z_predict - Z3))

# Optimizer
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train3 = optimizer3.minimize(loss3)

# Initialize variables
init3 = tf.global_variables_initializer()

# Run
with tf.Session() as sess:
    sess.run(init3)
    
    # Train
    for epoch in range(1000):
        sess.run(train3, {X3: x1, Y3: y1, Z3: z1})
        
    # Get the optimized parameter
    final_wz, final_bz = sess.run([Wz, bz])

# Plot the result
x1_test3 = np.linspace(20, 190, 10)
y1_test3 = final_wz*x1_test3 + final_bz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, c="b", marker="o", label="Designer Input")
ax.scatter(x1_test3, y1_test3, z1_test3, c="r", marker="o", label="3d Sketch Model")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

