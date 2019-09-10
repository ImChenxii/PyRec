import math
from sklearn.utils import shuffle


class Data4PyRec(object):
    def __init__(self, data_set, label, batch_size=128, is_shuffle=False, random_seed=2018):
        self.data_set = data_set # 原始DataFrame数据集
        self.label = label # 原始label
        self.batch_seq = 0 # 初始化batch序号
        self.batch_size = batch_size # 读取到指定的batch_size
        self.total_data_num = data_set.shape[0] # 总数据量的大小
        self.batch_num = math.floor(self.total_data_num / self.batch_size) # 总共batch数量(从0开始计,所以向下取整)
        if is_shuffle: # 是否将数据集随机打乱
            self.data_set = shuffle(self.data_set, random_state=random_seed).reset_index(drop=True)
            self.label = shuffle(self.label, random_state=random_seed).reset_index(drop=True)

    def next(self):
        self.batch_seq += 1
        batch_data = self.data_set[(self.batch_seq - 1) * self.batch_size, self.batch_seq * self.batch_size] # 不用考虑越界,Python没有越界异常
        batch_label = self.label[(self.batch_seq - 1) * self.batch_size, self.batch_seq * self.batch_size]
        return (batch_data, batch_label)

    def has_next(self):
        return self.batch_seq < self.batch_num

    def reset(self):
        self.batch_seq = 0

    def get_seq(self):
        return self.batch_seq







