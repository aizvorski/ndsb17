import multiprocessing
import os


def gpu_init(queue):
    global gpu_id
    gpu_id = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    process = multiprocessing.current_process()
    print(gpu_id, process.pid, gpu_list)

def f(x):
    global gpu_id

    import tf
    process = multiprocessing.current_process()
    return (gpu_id, process.pid, x * x)

manager = multiprocessing.Manager()
gpu_init_queue = manager.Queue()

num_gpu = 8
for i in range(num_gpu):
    gpu_init_queue.put(i)

p = multiprocessing.Pool(num_gpu, gpu_init, (gpu_init_queue,))
p.map(f, range(16))
