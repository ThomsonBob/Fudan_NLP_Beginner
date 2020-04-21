import torch
from torch.utils.data import Dataset

'''torch.utils.data    Dataset'''
class SnliDataSet(Dataset):
    def __init__(self,data,max_premise_len=None,max_hypothesis_len=None):
        #序列长度
        self.num_sequence=len(data["premise"])
        
        #创建tensor矩阵的尺寸
        self.premise_len=[len(seq) for seq in data["premise"]]
        self.max_premise_len=max_premise_len
        if self.max_premise_len is None:
            self.max_premise_len=max(self.premise_len)
        
        self.hypothesis_len=[len(seq) for seq in data["hypothesis"]]
        self.max_hypothesis_len=max_hypothesis_len
        if max_hypothesis_len is None:
            self.max_hypothesis_len=max(self.hypothesis_len)
#         print(self.num_sequence, self.max_premise_len)
#         print(self.num_sequence, self.max_hypothesis_len)
        #转成tensor，封装到data里
        self.data= {
            "premise":torch.zeros((self.num_sequence,self.max_premise_len),dtype=torch.long),
            "hypothesis":torch.zeros((self.num_sequence,self.max_hypothesis_len),dtype=torch.long),
            "labels":torch.tensor(data["labels"])
        }
        
        for i,premise in enumerate(data["premise"]):
            l=len(data["premise"][i])
            self.data["premise"][i][:l]=torch.tensor(data["premise"][i][:l])
            l2=len(data["hypothesis"][i])
            self.data["hypothesis"][i][:l2]=torch.tensor(data["hypothesis"][i][:l2])
        
        
    def __len__(self):
        return self.num_sequence
        
    def __getitem__(self,index):
        return { "premise": self.data["premise"][index],
                    "premise_len":min(self.premise_len[index], self.max_premise_len),
                    "hypothesis":self.data["hypothesis"][index],
                    "hypothesis_len":min(self.hypothesis_len[index], self.max_hypothesis_len),
                    "labels":self.data["labels"][index]   }