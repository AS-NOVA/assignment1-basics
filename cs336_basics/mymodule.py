import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, einsum, reduce, pack, unpack
from jaxtyping import Float, Int, Bool
import math

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

class RotaryPositionalEmbedding(nn.Module):
    cosines: torch.Tensor
    sines: torch.Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        # 初始化函数只需要负责根据输入的构建并存储起所有的正余弦值，便于使用
        super().__init__()
        assert d_k%2==0 , "dimension of Q and K should be even for RoPE"
        halfd = int(d_k / 2)

        positions = torch.arange(max_seq_len, device=device) # 序列位置参数：0~maxlen-1
        S = math.pow(theta,-2/d_k)
        thetas = torch.pow(S,torch.arange(halfd,device=device)) # theta的指数从0~大约-1，各分量的最小旋转角从1~1/theta都有
        thetas_with_position = einsum(positions,thetas,"maxlen, halfdk -> maxlen halfdk")

        self.register_buffer("cosines", 
                             torch.cos(thetas_with_position),   
                             persistent=False)
        self.register_buffer("sines", 
                             torch.sin(thetas_with_position),
                             persistent=False)
    
    def forward(self, 
                x: torch.Tensor, 
                token_positions: torch.Tensor) -> torch.Tensor:
        # 先把输入的最后一位按奇偶分开，分别进行乘法后重新进行线性组合
        # 使用sin和cos值时注意截断
        # x: (..., len, d_k)
        # token_positions: (..., len)
        #print("输入tensor维度：",x.shape)
        #print("输入位置索引维度：",token_positions.shape)
        rearranged_x = rearrange(x,"... len (halfdk two) -> ... len halfdk two",two=2)
        #print("")
        oddx = rearranged_x[...,1] # ... len halfdk
        evenx = rearranged_x[...,0] # ... len halfdk
        #print("输入按奇偶切分后维度：",oddx.shape)
        cut_cosines = self.cosines[token_positions]
        cut_sines = self.sines[token_positions]
        # 三角函数阵： halfdk         
        # rotated_oddx = einsum(oddx, cut_cosines , "... len halfdk, len halfdk -> ... len halfdk") + \
        #              einsum(evenx, cut_sines, "... len halfdk, len halfdk -> ... len halfdk")
        # rotated_evenx = einsum(evenx, cut_cosines, "... len halfdk, len halfdk -> ... len halfdk") - \
        #               einsum(oddx, cut_sines, "... len halfdk, len halfdk -> ... len halfdk")
        rotated_oddx = oddx * cut_cosines + evenx * cut_sines
        rotated_evenx = evenx * cut_cosines - oddx * cut_sines

        res = rearrange([rotated_evenx, rotated_oddx],
                        "two ... len halfdk -> ... len (halfdk two)")
        return res

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    xdim = len(x.shape)
    dim = dim % xdim
    pat_origin = " ".join([f"d{i}" for i in range(xdim)])
    pat_reduce = " ".join([f"d{i}" if i != dim else "1" for i in range(xdim)])
    x_max = reduce(x,f"{pat_origin}->{pat_reduce}","max")
    expx = torch.exp(x - x_max)
    sum_expx = reduce(expx,f"{pat_origin}->{pat_reduce}","sum")
    res = expx / sum_expx
    return res

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scaled_prod = einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys") / math.pow(d_k, 1/2)
                    
    if mask != None:
        scaled_prod.masked_fill_(~mask, -torch.inf)

    probs = softmax(scaled_prod,-1)
    res = einsum(probs, V, "... queries keys_also_values, ... keys_also_values d_v -> ... queries d_v")
    return res

class MultiHeadSelfAttention(nn.Module):
    # 这里多头注意力的算法参考原始transformer
    # 所以dk = dv = dmodel / heads
    def __init__(self, 
                 d_model:int,
                 num_heads:int,
                 max_seq_len:int,
                 use_rope:bool = False,
                 rope_theta:float = 10000.0,
                 token_positions:Int[Tensor, " ... sequence_length"] | None = None,
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.token_positions = token_positions
        parameter_kwargs = {"device":device, "dtype":dtype}

        if self.d_model % self.num_heads != 0:
            raise ValueError("num_heads cant divide d_model")
        
        self.d_head = int(self.d_model / self.num_heads)
        self.linear_Wq = Linear(self.d_model, self.d_model, **parameter_kwargs)
        self.linear_Wk = Linear(self.d_model, self.d_model, **parameter_kwargs)
        self.linear_Wv = Linear(self.d_model, self.d_model, **parameter_kwargs)
        self.linear_Wo = Linear(self.d_model, self.d_model, **parameter_kwargs)

        self.pre_cache_mask =torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()

        if self.use_rope == True:
            self.rope = RotaryPositionalEmbedding(self.rope_theta,
                                                  self.d_head,
                                                  self.max_seq_len,)

    def forward(self, 
                x:Float[Tensor,"... len d_model"]
                )->Float[Tensor,"... len d_model"]:
        len = x.shape[-2]
        Q = self.linear_Wq(x)
        K = self.linear_Wk(x)
        V = self.linear_Wv(x)
        Qs = rearrange(Q,"... len (num_heads d_head) -> ... num_heads len d_head", num_heads = self.num_heads)
        Ks = rearrange(K,"... len (num_heads d_head) -> ... num_heads len d_head", num_heads = self.num_heads)

        #对每个head进行相同的rope
        if self.use_rope == True:
            Qs = self.rope(Qs,self.token_positions)
            Ks = self.rope(Ks,self.token_positions)

        Vs = rearrange(V,"... len (num_heads d_head) -> ... num_heads len d_head", num_heads = self.num_heads)
        mask = self.pre_cache_mask[:len,:len]
        As = scaled_dot_product_attention(Qs,Ks,Vs,mask)
        A = rearrange(As,"... num_heads len d_head -> ... len (num_heads d_head)")
        res = self.linear_Wo(A)
        return res        

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,)->None:
        super().__init__()

