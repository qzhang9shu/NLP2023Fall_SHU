import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

train_data_path = "./mnist/train"
test_data_path = "./mnist/test"
mnist_ds = ds.MnistDataset(train_data_path)
dic_ds = mnist_ds.create_dict_iterator()

def get_data():
    train_data_path = "./mnist/train"
    test_data_path = "./mnist/test"
    mnist_ds = ds.MnistDataset(train_data_path)
    dic_ds = mnist_ds.create_dict_iterator()
    return  train_data_path,test_data_path

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    mnist_ds = ds.MnistDataset(data_path)
    # 定义数据增强和处理所需的一些参数
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 根据上面所定义的参数生成对应的数据增强方法，即实例化对象
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 将数据增强处理方法映射到（使用）在对应数据集的相应部分（image，label）
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 处理生成的数据集
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

def dataset(train_data,test_data):
    ms_dataset = create_dataset(train_data_path)
    eval_dataset = create_dataset(test_data_path)
    return ms_dataset,eval_dataset