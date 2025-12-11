import numpy.typing as npt
import numpy as np
import torch
from typing import BinaryIO, IO
import os


# uv run pytest -k test_get_batch

def my_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从numpy数组格式的dataset中随机选择起点，采样batch_size个序列，每个序列长度为context_length。
    Args:
        dataset (npt.NDArray): 一维numpy数组，表示token序列
        batch_size (int): 批量大小
        context_length (int): 上下文长度
        device (str): 设备字符串，如'cpu'或'cuda'
    Returns:
        output(tuple[torch.Tensor, torch.Tensor]):
            - seqs: 形状为(batch_size, context_length)的tensor，表示输入序列
            - labels: 形状为(batch_size, context_length)的tensor，表示标签序列
            - seqs 和 labels 相互错开一位
    """
    data_len = len(dataset)
    start_pos = np.random.randint(0,data_len-context_length,batch_size)
    # 起点下标下限为0，上限为data_len-context_length-1
    # 最靠后的序列末端下标为data_len-2
    # 最靠后的序列末端对应的label下标为data_len-1
    # 正好就是data_len的最后一个下标
    seqs = [dataset[start:start+context_length] for start in start_pos]
    labels = [dataset[start+1:start+context_length+1] for start in start_pos]
    seqs_np = np.stack(seqs)
    seqs_tensor = torch.from_numpy(seqs_np).to(device=device)
    labels_np = np.stack(labels)
    labels_tensor = torch.from_numpy(labels_np).to(device=device)
    return seqs_tensor,labels_tensor





def my_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def my_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']