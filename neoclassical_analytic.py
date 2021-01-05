## By SeHyoun Ahn, Jan 2021
## MIT-license

## Uncomment the next two lines if you want to use GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



import tensorflow as tf
import matplotlib.pyplot as plt


# Set seed for replicability
seed_in = 42
tf.random.set_seed(seed_in)
g1 = tf.random.Generator.from_seed(seed_in)



##
# Define Paramters
gamma = 2
rho = 0.04
A = 0.5
alpha = 0.36
delta = 0.05

k_min = 0.2 # must be smaller than 1.8
k_max = 10 # must be bigger than 4
n_sample = 1000
dV_cutoff = 1e-4

crit_val = 1e-10
num_epochs = 100000

optimizer = tf.keras.optimizers.Adam()



## Define Neural Network
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(8, activation='tanh', input_shape=(1,)),
  tf.keras.layers.Dense(8, activation='tanh'),
  tf.keras.layers.Dense(8, activation='tanh'),
  tf.keras.layers.Dense(1)
])



## Define Loss Function for HJB
@tf.function
def loss(model, x, training):
  with tf.GradientTape(persistent=True) as tape_var:
    # Compute dV/dx
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      V_pred = model(x)
    dV_dx = tape.gradient(V_pred, x)

    V = model(x)
    c = tf.math.maximum(dV_dx, dV_cutoff)**(-1/gamma) # c from analytic expression

    u = (c**(1-gamma))/(1-gamma)
    y = A*(x**alpha)

    # HJB equation
    error_hjb = rho*V - u - dV_dx*(y - delta*x - c)
    # error_hjb = tf.where(tf.math.is_nan(error_hjb), 10.0, error_hjb)
    error_hjb = tf.math.reduce_mean(tf.math.square(error_hjb))

    # Supplementary condition that dV/dx > 0
    error_dV = tf.where(dV_dx < dV_cutoff, dV_dx-dV_cutoff, 0)
    error_dV = tf.math.reduce_mean(tf.math.square(error_dV))

    # Boundary condition below
    error_bd1 = tf.where((x<k_min+1) & (y - delta*x - c < 0), c*10.0, 0)
    error_bd1 = tf.math.reduce_mean(tf.math.square(error_bd1))

    # Boundary condition above
    error_bd2 = tf.where((x>k_max-1) & (y - delta*x - c > 0), 10.0/c, 0)
    error_bd2 = tf.math.reduce_mean(tf.math.square(error_bd2))

    loss_value = error_hjb + error_dV + error_bd1 + error_bd2

  grad = tape_var.gradient(loss_value, model.trainable_variables)
  return (loss_value, grad)



## Train the Neural-Network
train_loss_results = []
for epoch in range(num_epochs):
  # Generate Test points
  x = g1.uniform([n_sample, 1], minval=k_min, maxval=k_max)

  # Compute loss and gradients
  loss_value, grads = loss(model, x, training=True)

  # gradient descent
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss_results.append(loss_value)
  if loss_value < crit_val:
    break
  if epoch % 100 == 0:
    print("Epoch {:07d}: Loss: {:.7f}".format(epoch, loss_value))
  if epoch % 1000 == 0:
    ## Diagonostic plots
    x = g1.uniform([n_sample, 1], minval=k_min, maxval=k_max)
    V = model(x)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      V_pred = model(x)

    dV_dx = tape.gradient(V_pred, x)

    V = model(x)

    c = tf.math.pow(tf.math.maximum(dV_dx, 1e-4), -1/gamma)
    u = (c**(1-gamma))/(1-gamma)
    y = A*(x**alpha)
    error_hjb = rho*V - u - dV_dx*(y - delta*x - c)

    plt.scatter(x, V)
    plt.title('Value Function')
    plt.xlabel('capital (k)')
    plt.savefig('V.png')
    plt.clf()

    plt.scatter(x, dV_dx)
    plt.title('dV/dx')
    plt.xlabel('capital (k)')
    plt.savefig('dV.png')
    plt.clf()

    plt.scatter(x, c)
    plt.title('consumption (c)')
    plt.xlabel('capital (k)')
    plt.savefig('c.png')
    plt.clf()

    plt.scatter(x, u)
    plt.title('utility (u)')
    plt.xlabel('capital (k)')
    plt.savefig('u.png')
    plt.clf()

    plt.scatter(x, y)
    plt.title('production k^a')
    plt.xlabel('capital (k)')
    plt.savefig('y.png')
    plt.clf()

    plt.scatter(x, (y - delta*x - c))
    plt.title('savings')
    plt.xlabel('capital (k)')
    plt.savefig('savings.png')
    plt.clf()

    plt.scatter(x, error_hjb)
    plt.title('error')
    plt.xlabel('capital (k)')
    plt.savefig('error.png')
    plt.clf()

    plt.plot(train_loss_results)
    plt.yscale('log')
    plt.title('Training metric')
    plt.savefig('training.png')
    plt.clf()
