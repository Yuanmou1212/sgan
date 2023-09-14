import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):  # 输入值是getitem 返回多次后对应元素构成的列表， 在这里经行堆叠，成一个batch的数据
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list] #len 返回第一个维度的长度 即行人数量， 构成list，每个元素就是每个scene中行人的数量
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)  # 还是在人数这个维度上堆叠。
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)  
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f: # r read only #文件对象 f 可以迭代文件的内容，使您能够逐行读取文件，处理每一行的数据
        for line in f:
            line = line.strip().split(delim)  # strip() 方法用于去除文本行两端的空白字符，包括空格、制表符、换行符等 split（）eg：文本行 "John\t30\tNew York" 将被分割成 ["John", "30", "New York"]
            line = [float(i) for i in line] # list line 中的每个元素转化成float
            data.append(line)
    return np.asarray(data) # numpy array


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] # 2 2阶多项式，返回残差来分辨拟合的如何。
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir) # list，包含该目录下 所有文件和子目录的名称
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # 所有文件的路径
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)  # 自定义的 return numpy array， 元素是float
            frames = np.unique(data[:, 0]).tolist()  # 提取了 data 列表中第0列的所有不重复的数值，构成list。
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :]) # bool 运算，每个unique的frame 回去对应data的数据，从而把frame_id 一样的放到一起，整个frame_data 按照frame大小从前往后
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip)) # N个球，M个连续球一组，有多少组的问题：减去最后一组，前面的所有球都可以是M球一组中第一个球，+1加上最后一组。/skip就是一步算还是多步一算。

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate( # 输入是个list，所以确实在拼接；
                    frame_data[idx:idx + self.seq_len], axis=0) #[start:stop] stop不包含在切片中
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 提取唯一的行人ID，因为这种情况常见：同一个frame下同一个行人多次出现
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))  # sequence中有多少人（一个dim） x和y数据（一个dim）时序 （一个dim）
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq): # 该sequence下所有行人ID，对每个行人单独处理。
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # 又是bool 挑出该行人的“行”，做slice，切出改行人的数据
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4) # 保留到小数点后4位，简化data
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # idx当前序列的起始位置。 frames（就是个list）的index， 根据当前行人序列的第一行数据的frameID 在 frames 列表中的索引位置
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # 最后一行的帧id 在frames列表的位置。
                    if pad_end - pad_front != self.seq_len:   # 任何时候发现目前sequence下的这个frame里的人，data不够seq这么长，则跳过这个人的处理，经行下一个人的处理。
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # dim2 上 0，1 代表frame和 ped_id 后面的才是位置数据等。 转置后：行代表x，y这些（比如第一行是x）；列代表不同frame，就是时间步
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative  相对坐标（相对于上一个时间步的变化）
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # 现在列代表不同的frame，（时刻），从2列到最后 - 从1列到导数第二个。就是现在-前一刻=坐标变化。
                    _idx = num_peds_considered  # 当前行人在序列中的索引位置
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq    # pad_front:pad_end 由于计算时-idx，所以是始终0到seq_len. 
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold)) # 自定义
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])  # 只有被选出来的行人，对应的参数才是1，算loss时候才会计算。
                    seq_list.append(curr_seq[:num_peds_considered]) # 把没有填充的部分，去掉。 因为有的人，不在scene足够长的时间，所以会被去掉
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])  #  numpy（ped_idx, position(x,y）, frame (time)) 就是一个scene！，n个人，一段时间，的位置。

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)  # 从第0维度，人数， 拼起来，再通过每个scene 不同人数的记录，后面还能对应拆开这个维度，得到不同的scene
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)  # 好处是 因为不同scene 的ped人数不一样，所以这个维度拼可以拼。但是如果加个维度，也只是代表不同scene， 还是得沿着人数的维度拼接如果想拼成一个tensor的话。
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)   # sequence 中 一部分是obs用于train 一部分是test
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # num_peds_in_seq 表示每个sequence （scene）中有多少（纳入计算）人。
        self.seq_start_end = [  # 而cumsum 是对list滚动求和，第二项=原1项和2项之和， 第三项=原1，2，3项只和... 用来算index  ； [0] 就是为了第一项是0
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:]) # end就是下一个项的起点。
        ]  # 这么做有效的原因也是 他前面 拼接的时候 concatenate 把不同scene的 人拼在一起了，np.concatenate(seq_list, axis=0) 所以要这样操作。

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]  # 因为人数始终是第一维度。
        ]
        return out
