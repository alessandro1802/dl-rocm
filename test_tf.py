import tensorflow as tf

print("TF version:", tf.__version__)

def configureGPUmemory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)

#configureGPUMemory()

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Matmul test:")
tf.debugging.set_log_device_placement(True)
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

print("Training test:")
#TODO
