import tensorflow as tf

# TensorFlow 2.x 中检查可用的 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 TensorFlow 使其在发现 GPU 时尽可能多地使用 GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Num GPUs Available: {len(gpus)}")
    except RuntimeError as e:
        # 在程序启动时设置 GPU 显存增长必须在程序初始化时完成
        print(e)
else:
    print("GPU is not detected")

# TensorFlow 2.x 中的计算无需 Session
a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b
print(c.numpy())  # 直接使用 .numpy() 方法获取计算结果



