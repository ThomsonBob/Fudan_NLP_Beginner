"""
Utility functions for the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x embedding_dim).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index =sequences_lengths.sort(0, descending=descending)
    #对sequence_lengths排序，返回一个排序数组和一个原数组下标排序后的顺序
    sorted_batch = batch.index_select(0, sorting_index)
    #按照sorting_index重新选择排序原来的批数据
    idx_range =sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    #创建一个新的和原来的sequences_lengths一样长的数组，好像顺序是从零开始的？
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    #reverse_mapping的数组排序就是squences_lengths在排序后数组的位置
    restoration_index = idx_range.index_select(0, reverse_mapping)
    #restoration_index好像是和mapping一样？
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    #reshaped_tensor为一个二位张量，保持第二维不变
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    # Reshape the mask so it matches the size of the input tensor.
    reshaped_mask = mask.view(-1, mask.size()[-1])  #这样不会有问题？
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)     #批矩阵乘法
    while mask.dim() < weighted_sum.dim(): 
        mask = mask.unsqueeze(1)   
    mask = mask.transpose(-1, -2)  #交换最后两个维度
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask

def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences.

    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).

    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0] 
    max_length = torch.max(sequences_lengths) #找到最长的sequence长度
    mask = torch.ones(batch_size, max_length, dtype=torch.float) 
    mask[sequences_batch[:, :max_length] == 0] = 0.0 
    #将所有batch中所有的语句结束之后的都变成零，使得梯度恒为零 
    return mask


def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add