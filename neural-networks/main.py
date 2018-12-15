import tensorflow as tf

if sys.version_info[0] != 2 or sys.version_info[1] != 7:
	raise SystemExit('Please use Python version 2.7.')

tf.enable_eager_execution()
stuff = tf.random_normal([1000, 1000])
more = tf.reduce_sum(stuff)
print(more)

