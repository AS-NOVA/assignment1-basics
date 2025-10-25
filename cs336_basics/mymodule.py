import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int

class Linear(nn.Module) :
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None
                 ) -> None:
        super().__init__()
        # in out维度记录在模块内部
        # tensor位置和类型信息并不属于模块，而是跟着tensor走
        # 所以只是传给创建tensor的函数，需要这些信息时直接问tensor而非模块
        self.in_features = in_features
        self.out_features = out_features
        parameter_kwargs = {"device": device, "dtype": dtype}

        # 创建一块参数矩阵
        # 由于pytorch存储参数时是按行的，每行内容都会连在一起
        # 而进行乘法时一定是长为in_features的那一维去乘输入向量
        # 所以一定是让in_features作为行的长度，out_features作为行的数量
        self.W = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                **parameter_kwargs    # 注意输入的设备和类型信息传到了这里！
            ))
        
        # 初始化：方差为2/(d_in + d_out)，截断处在3个标准差
        var = 2.0 / (in_features + out_features)
        std = var ** 0.5
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x的形状是 (..., d_in)
        # W的形状是 (d_out, d_in)
        # 输出形状是 (..., d_out)
        # 简单写法：return x @ self.W.T
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")   #和上面的等价
        # 注意：x前面可能有许多维度，但最后一维一定是输入维度d_in
        # 而为了方便计算，我们的W是d_out*d_in的
        # 所以正常需要转置W才能相乘！这里用einsum来指定怎么乘，可以避免手动转置
        
class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings:int, # 词表大小
                 embedding_dim:int, # 隐藏空间大小
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None
                 ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        parameter_kwargs = {"device": device, "dtype": dtype}
        self.embedding_matrix = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                **parameter_kwargs
            )
        )
        nn.init.trunc_normal_(self.embedding_matrix, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # tensor操作：直接用tensor作为索引！
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device:torch.device | None = None, 
                 dtype:torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        parameter_kwargs = {"device": device, "dtype": dtype}

        self.gain_parameter = nn.Parameter(
            torch.empty(
                d_model,
                **parameter_kwargs
            )
        )

        nn.init.ones_(self.gain_parameter)

        # 其实可以用torch.ones直接初始化，不过这里拆成两部分便于理解


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2,"... d -> ... 1","mean") + self.eps) ** -1 
        # 将x的最后一维压缩为一个标量作为分母，但是保留这一维度，便于广播
        res = x * rms * self.gain_parameter
        # 第一个乘法是因为手动保留了维度所以才能进行的，第二个乘法会自动广播
        return res.to(in_dtype)

def swish(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLUFFN(nn.Module):
    def __init__(self, 
                 d_model:int,
                 d_ff:int,
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        parameter_kwargs = {"device":device,"dtype":dtype}

        self.linear1 = Linear(d_model,d_ff,**parameter_kwargs)
        self.linear2 = Linear(d_ff,d_model,**parameter_kwargs)
        self.linear3 = Linear(d_model,d_ff,**parameter_kwargs)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:   
        activates = swish(self.linear1(x))
        gates = self.linear3(x)
        return self.linear2(activates * gates)
    


