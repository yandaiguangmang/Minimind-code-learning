# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__( #参数的初始化，超参数设置
            self,
            dropout: float = 0.0, #随机舍弃的概率
            bos_token_id: int = 1, #句子开始特殊token的id
            eos_token_id: int = 2, #句子结束特殊token的id
            hidden_act: str = 'silu', # 激活函数为silu函数
            hidden_size: int = 512, #隐藏层维度大小
            intermediate_size: int = None, #FFN的中间层维度，即升维要升到的维度，通常设为隐藏层维度的8/3倍且向上取整为64的倍数
            max_position_embeddings: int = 32768, #模型支持的最大序列长度
            num_attention_heads: int = 8, #注意力头数
            num_hidden_layers: int = 8, #隐藏层数即transformer层数
            num_key_value_heads: int = 2,#共享KV的组数
            vocab_size: int = 6400, #词表大小
            rms_norm_eps: float = 1e-05,#rmsnorm中防止除0
            rope_theta: int = 1000000.0, #rope的参数
            inference_rope_scaling: bool = False,#是否在推理时应用 YaRN 缩放方法扩展上下文窗口。若为 True，则 rope_scaling 字典生效
            flash_attn: bool = True, #是否启用flash attention
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False, 
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs) #调用父类初始化方法，并赋值
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None #启用外推，用于将训练时的 2048 长度扩展到推理时的 32768
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):#输入归一化层
    def __init__(self, dim: int, eps: float = 1e-5):#类的初始化，规定维度的参数类型、设置防止除0的参数值
        super().__init__()#继承超类的初始化函数
        self.eps = eps#防止除0的参数设置
        self.weight = nn.Parameter(torch.ones(dim))#将权重设置为全1的张量，维度和输入一致

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)#先将x平方，再对其进行均分（保持维度不变），再加上防止除0的数值，最后对这个整体进行开方取倒数

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)#将经过norm函数的值乘于输出x和权重值得到最终的输出y


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,#rope+yarn
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0#计算rope的频率i，其中底数为rope_base,指数为从0开始一直到dim/2（向下取整）以2（维度划分两两一组做旋转）为步长的值除以维度
    if rope_scaling is not None:#需要计算yarn
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )#从已经定义好初始化好的超参数中取出yarn所需参数值，若未能取出则为其赋值
        if end / orig_max > 1.0:#当推理的长度超过了训练时的最大长度，需要应用yarn进行外推
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))#给定一个缩放因子 b，返回 RoPE 的某个维度索引 i，使得该维度对应的波长恰好等于orig_max/b : i=(dim/2)*ln(org_max/2pai*b)/ln(rope_base)
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)#带入β_low、β_fast计算缩放区间的上下界，max\min确保在索引范围内
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)#构建斜坡函数γ(i)对索引i定义其插值权重：（i-low）/(high-low),使这个比值限制在0~1之间，即若i>hign为1，i<low为0
            freqs = freqs * (1 - ramp + ramp / factor)#应用yarn调整频率：f'(i) = f(i)((1-γ) + γ/s)

    t = torch.arange(end, device=freqs.device)#生成位置索引1、2、3...
    freqs = torch.outer(t, freqs).float()#外积，构建(end,dim//2)规模的二维张量,即计算每个位置每个频率的角度mθ_i：[θ_0,θ_1,θ_2,... ]
                                                                                                           #[2*θ_0,2*θ_1,2*θ_2,... ]                    
                                                                                                           #[3*θ_0,3*θ_1,3*θ_2,... ]
                                                                                                           #....

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor #对角度取余弦并在最后一个维度进行拼接，得到完整的旋转矩阵（一对共享一个，将其写成两个一样的）[θ_0,θ_1,θ_2,... θ_0,θ_1,θ_2,... ]
                                                                                                                                                                          #[2*θ_0,2*θ_1,2*θ_2,...2*θ_0,2*θ_1,2*θ_2,...]                    
                                                                                                                                                                          #[3*θ_0,3*θ_1,3*θ_2,...3*θ_0,3*θ_1,3*θ_2,...]
                                                                                                                                                                          #....







    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor#取正弦，同理
    return freqs_cos, freqs_sin #返回余弦、正弦值


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):#unsqueeze：在cos\sin的第一维度上扩展维度，因为q\k张量为三维，为了匹配维度。q、k张量最后一维往往前半部分是实部，后半部分为虚部，实际上(i,i+dim//2)是一组待旋转的向量
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)#将x张量的最后一维从中间分开，[a,b]->[-b,a],将后半部分取负旋转到前面

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))#先将cos\sin扩展维度：(seq_len, head_dim)，通过 unsqueeze（默认unsqueeze_dim=1）变为 (seq_len, 1, head_dim)，再进行q与cos逐元素相乘（由于广播机制(seq_len, 1, head_dim) 自动扩展为 (batch_size, seq_len, num_heads, head_dim)（复制 batch 和 head 维度）），再用旋转取负后的q与sin逐元素相乘再相加，得到旋转后的q_embed.
    #例如：
    #head_dim = 4，则 x = [a, b, c, d]。
    #rotate_half(x) = [-c, -d, a, b]。
    #用 cos = [cosθ₀, cosθ₁, cosθ₀, cosθ₁] 和 sin = [sinθ₀, sinθ₁, sinθ₀, sinθ₁] 计算：
    #x_embed[0] = a*cosθ₀ + (-c)*sinθ₀；embed[1] = b*cosθ₁ + (-d)*sinθ₁；x_embed[2] = c*cosθ₀ + a*sinθ₀；x_embed[3] = d*cosθ₁ + b*sinθ₁ .得到的向量前半部为实部，后半部为对应虚部。
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))#k的处理也同理
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor: #在GQA中的query组中“复制”多份k\v实现共享，kv cache中实际上只有一个副本，进行重复实现计算attention时共享
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape #将共享的k\v张量各维度的形状赋值给局部变量
    if n_rep == 1: #如果只有一组只有一个query说明退化成多头注意力，直接返回即可
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)#先在num_key_value_heads, head_dim之间插入一个新维度，再对这个维度扩展到重复的次数，实际上并不会直接分配内存只是这几个维度指向一个地址，最后进行reshape打平让第三维等于query的头数
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads #对k\v的头数即组数进行赋值，如果参数中取出来是空则赋值为attention的头数
        assert args.num_attention_heads % self.num_key_value_heads == 0  #attentiion的头数必须能整除k\v的头数即组数，才能确保恰好将q分到每一组
        self.n_local_heads = args.num_attention_heads  #赋值
        self.n_local_kv_heads = self.num_key_value_heads #赋值
        self.n_rep = self.n_local_heads // self.n_local_kv_heads #k\v在计算attention时需要重复的次数，即每一组内q的个数等于总的注意力头数向下除以组数
        self.head_dim = args.hidden_size // args.num_attention_heads  #每一个注意力头的维度，即总的隐藏层维度向下除以总的注意力头数
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False) #输入x变成query经过的投影层，实际上是乘于一个投影矩阵，输入维度为隐藏层维度，输出维度attention头数乘于每一个头的维度
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False) #输入x变成key经过的投影层，与上同理
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False) #输入x变成value经过的投影层，与上同理
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False) #attention计算完后的输出通过的线性层，将注意力输出（拼接后的多头结果）映射回隐藏层大小。，输入维度和输出维度与上对调
        self.attn_dropout = nn.Dropout(args.dropout) #dropout层，防止过拟合
        self.resid_dropout = nn.Dropout(args.dropout) #残差连接层，保留之前特征
        self.dropout = args.dropout #为flash attention准备的丢弃概率参数
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn#检查是否支持scaled_dot_product_attention 且启用 Flash Attention，用于加速训练和推理
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor, #输入张量，形状 (batch_size, seq_len, hidden_size）
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin。包含 (cos, sin) 的元组，由 RoPE 预计算生成，形状均为 (seq_len, head_dim)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, #可选的 KV 缓存，用于增量推理。格式为 (past_key, past_value)，形状为 (batch_size, past_len, num_kv_heads, head_dim)
                use_cache=False, #是否返回更新后的 KV 缓存
                attention_mask: Optional[torch.Tensor] = None): #可选的注意力掩码，形状通常为 (batch_size, seq_len)，值为 0/1。
        bsz, seq_len, _ = x.shape #输入各维度形状赋值
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) #对输入进行线性投影
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)#对q进行形状重塑为四维张量，便于后续处理
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)#对k进行形状重塑为四维张量，便于后续处理
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)#对v进行形状重塑为四维张量，便于后续处理

        cos, sin = position_embeddings #赋值正余弦值
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin) #调用之前的应用位置编码函数将旋转位置信息注入query和key

        # kv_cache
        if past_key_value is not None: #如果提供了 past_key_value（即当前为解码阶段，已生成部分 token），则将历史 KV 与当前步的 KV 拼接起来，使 seq_len 扩展为 past_len + 1
            xk = torch.cat([past_key_value[0], xk], dim=1)#第一维度进行拼接k，即序列长度维度
            xv = torch.cat([past_key_value[1], xv], dim=1)#同理拼接v
        past_kv = (xk, xv) if use_cache else None #若 use_cache=True，则将缓存更新为拼接后的完整 KV，供下一轮使用.

        xq, xk, xv = (
            xq.transpose(1, 2),#将形状从 (bsz, seq_len, num_heads, head_dim) 变为 (bsz, num_heads, seq_len, head_dim)，即变成多批的小头进行处理
            repeat_kv(xk, self.n_rep).transpose(1, 2),#调用上面的repeat函数将 key的头重复 n_rep 次，使头数与 query 一致，实现组内共享kv并行化进行GQA计算。然后同样进行转置。
            repeat_kv(xv, self.n_rep).transpose(1, 2) #对value同理操作
        )#最终三者的形状均为 (bsz, n_local_heads, seq_len, head_dim)，其中 n_local_heads = n_local_kv_heads * n_rep

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):#若self.flash 为 True；seq_len > 1，避免单步时潜在的问题；past_key_value is None：即非增量解码阶段 ；没有提供掩码，或提供的掩码全为 1（即无填充，无需掩码）
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True) #使用flash attention：训练时开启dropout，若是推理时关闭；自动进行掩码操作
        else:#不满足上述条件手动进行attention计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) #对k交换后两个维度即做矩阵转置与q点积并缩放得到注意力分数矩阵，形状(bsz, n_heads, seq_len, seq_len)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)#对最后seq_len个位置的分数加上上三角的-inf，确保当前位置只能看到之前的位置（包括自身，因为对角线上为 0，从 diagonal=1 开始置-inf）。这里用 -seq_len取最后一维的最后seq_len个元素: 是因为可能拼接了过去的 KV，过去的信息无需被掩码需要仅对当前序列部分施加因果掩码。

            if attention_mask is not None:#若提供了attention_mask，遮盖无效位置，不计算loss的位置例如填充token的位置
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #扩展维度，(batch_size,seq_len)->(batch_size, 1, 1, seq_len)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9#将有效位置于0，掩盖处置为1，从而将掩盖位置设为非常大的负数，归一化后变成0
                scores = scores + extended_attention_mask #将扩展的掩码矩阵加到注意力分数矩阵上（掩码矩阵自动广播）

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)#对最后一维应用softmax的注意力权重
            scores = self.attn_dropout(scores) #对注意力权重应用dropout,提高泛化能力
            output = scores @ xv #将注意力权重对value进行加权求和，得到输出，形状为(bsz, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)#先颠倒1和2维恢复到(bsz, seq_len, n_heads, head_dim)，再自适应拼接所有头得到(bsz, seq_len, n_heads * head_dim)
        output = self.resid_dropout(self.o_proj(output)) #先通过输出线性层映射回隐藏层大小，再应用残差dropout恢复特征
        return output, past_kv #返回GQA最终输出和kv cache


class FeedForward(nn.Module): #FFN模块
    def __init__(self, config: MiniMindConfig):#初始化函数，config 参数类型为 MiniMindConfig，包含模型配置信息，如隐藏层维度 hidden_size、激活函数类型 hidden_act、dropout 概率等
        super().__init__()  #继承父类的初始化函数
        if config.intermediate_size is None: #如果没有给定升维的维数，需要自己计算：
            intermediate_size = int(config.hidden_size * 8 / 3) #先对隐藏层维度乘于大约2.67倍
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64) #然后将计算结果向上取整到 64 的倍数，方便gpu进行优化
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) #输入的x进入激活函数作为门控信号前进行线性变化升维所需要的线性层，不加偏置
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)# 对输出的进行降维的线性层，不加偏置
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)# 对输入进行升维的线性层，不加配置
        self.dropout = nn.Dropout(config.dropout) #dropout层，为了增强泛化能力
        self.act_fn = ACT2FN[config.hidden_act] #激活函数SiLU，通过act2n将名字映射为真实函数

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))#输入x升维后的信息和输入x先线性变化升维再进入SiLU激活函数变成的门控信号逐元素相乘得到输出信号，对其进行降维回隐藏层维度最后dropout进行正则化防止过拟合。


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module): #transformer block,将GQA层和FFN层合并起来
    def __init__(self, layer_id: int, config: MiniMindConfig):# 初始化函数，传入配置参数
        super().__init__() #继承父类初始化方法
        self.num_attention_heads = config.num_attention_heads #注意力头数配置参数赋值
        self.hidden_size = config.hidden_size #隐藏层维度大小配置参数赋值
        self.head_dim = config.hidden_size // config.num_attention_heads #每一个注意力头的维度
        self.self_attn = Attention(config) #已经实现的GQA层

        self.layer_id = layer_id #layer的id号赋值
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) #输入到GQA时的RMSNorm归一化层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) #attention层的输出当作FFN层的输入时的RMSNorm归一化层
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config) #若未使用专家网络则赋值为FFN层，使用则为使用混合专家前馈网络

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):#继承nn必须实现的前向传递函数
        residual = hidden_states #保存输入作为加入残差的原始特征
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )#数据经过GQA即attention层得到输出值和KV cache的值
        hidden_states += residual #加入之前保存的输入，即应用残差网络
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) #经过FFN层后并加上残差的输出值
        return hidden_states, present_key_value #返回参数


class MiniMindModel(nn.Module):#组装各层
    def __init__(self, config: MiniMindConfig):#初始化函数
        super().__init__() #继承父类的初始化方法
        self.config = config #配置参数赋值
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers #词表大小、隐藏层数目按配置赋值
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) #embedding层设置
        self.dropout = nn.Dropout(config.dropout) #dropout层设置
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])# transformer(decoder-only)多层堆砌，从而打造深层网络
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) #RMSNorm层

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)# rope位置编码预计算旋转所需的cos\sin值
        self.register_buffer("freqs_cos", freqs_cos, persistent=False) #将cos值注册为模型的缓冲区，不参与梯度更新
        self.register_buffer("freqs_sin", freqs_sin, persistent=False) #将sin值注册为模型的缓冲区，不参与梯度更新

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, #输入经过分词器的token id序列
                attention_mask: Optional[torch.Tensor] = None, #可选掩码
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, #可选 KV 缓存列表，用于增量推理。每个元素对应一层的缓存 (key, value)，形状均为 (batch_size, past_len, num_kv_heads, head_dim)。
                use_cache: bool = False, #是否返回更新后的缓存
                **kwargs): #接受任意参数为了兼容
        batch_size, seq_length = input_ids.shape #批次大小、序列长度的赋值
        if hasattr(past_key_values, 'layers'): past_key_values = None #如果传入的是带有 layers 属性的对象（可能来自某些库的格式），则重置为 None；
        past_key_values = past_key_values or [None] * len(self.layers) #若没有提供缓存，则初始化为与层数等长的 None 列表。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 #start_pos 表示已经生成的 token 数量（即历史序列长度）。如果第一层的缓存不为空，则取 key 的第二维（past_len）作为起始位置；否则从 0 开始。该值用于从预计算的 RoPE 数组中切片正确的连续位置。

        hidden_states = self.dropout(self.embed_tokens(input_ids))#对token id序列进行embedding层的转化为稠密向量，再dropout正则化防止过拟合

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )#对rope预计算的正余弦值从当前序列开始的位置一直切片到当前输入序列末，只对当前处理的序列进行位置编码

        presents = [] #缓存
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)): #逐transformer层遍历
            hidden_states, present = layer( #每层接收当前 hidden_states、位置编码、该层的缓存、use_cache 标志和注意力掩码
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            ) #层内执行自注意力和前馈网络，返回更新后的 hidden_states 和本层的缓存 present（如果 use_cache=False 则为 None）
            presents.append(present) #将 present 添加到 presents 列表，最终返回所有层的缓存

        hidden_states = self.norm(hidden_states) #对最后一层的输出进行RMSNorm层归一化处理

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze()) #遍历所有层，如果该层的前馈网络是 MOEFeedForward（混合专家模块），则收集其 aux_loss 属性（通常用于专家负载均衡的损失），并求和。初始值 hidden_states.new_zeros(1).squeeze() 确保求和操作在正确的设备上进行，且结果为标量。
        return hidden_states, presents, aux_loss #返回最终输出，形状 (batch, seq_length, hidden_size)，可用于计算下一个 token 的 logits；presents：各层的 KV 缓存列表，供后续增量推理使用；aux_loss：MoE 辅助损失（若无 MoE 则为 0），在训练时与主损失相加。


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):#PreTrainedModel：Transformers库的基类，提供了预训练模型的通用方法。GenerationMixin：为模型添加生成文本所需的方法

    config_class = MiniMindConfig #config_class：指定该模型对应的配置类为 MiniMindConfig，这样在调用 from_pretrained 时能自动加载正确的配置。

    def __init__(self, config: MiniMindConfig = None):#初始化
        self.config = config or MiniMindConfig()#配置处理：若未传入 config，则创建一个默认的 MiniMindConfig 实例，并赋值给 self.config。
        super().__init__(self.config) #调用父类初始化
        self.model = MiniMindModel(self.config)#实例化 MiniMindModel（前面解析过的decoder-only Transformer），保存在 self.model
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) #最后的一个线性层，将 hidden_size 维的表示映射回词汇表大小，用于预测下一个 token 的概率分布。这里不使用偏置（bias=False）
        self.model.embed_tokens.weight = self.lm_head.weight #权重参数共享，将最后unembedding的线性层和embedding词嵌入层参数共享（指针指向而非深拷贝），减少权重参数数量和训练量，有利于训练稳定性

    def forward(self, #前向传播函数
                input_ids: Optional[torch.Tensor] = None, #经过分词器后输入的token id序列
                attention_mask: Optional[torch.Tensor] = None,#掩码矩阵
                labels: Optional[torch.Tensor] = None, #用于计算损失的目标 token IDs
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, #可选，KV 缓存列表，用于增量推理。
                use_cache: bool = False,#是否返回更新后的缓存
                logits_to_keep: Union[int, torch.Tensor] = 0,#控制返回的 logits 长度，可以是整数或张量切片。用于生成时只保留最后几个位置的 logits 以节省内存（例如在自回归生成中，只需最后一个 token 的 logits）。若为整数 k，则返回最后 k 个 token 的 logits；若为张量，则直接作为切片索引。
                **args):#接收额外参数，兼容
        hidden_states, past_key_values, aux_loss = self.model( #调用核心模型计算出最后一层输出、各层的kv cache、moe损失
            input_ids=input_ids, #输出的id序列
            attention_mask=attention_mask, #掩码
            past_key_values=past_key_values,#kv cache列表
            use_cache=use_cache, #是否更新kv cache
            **args#额外参数
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep #如果控制返回logits长度参数是整数，则创建切片只保存最后几个位置的隐藏状态；若已经为张量切片直接使用
        logits = self.lm_head(hidden_states[:, slice_indices, :]) #对hidden_states 按批次和切片索引在第二维即序列长度维度选取对应位置的隐藏状态，再通过 lm_head 得到 logits，形状为 (batch_size, kept_seq_len, vocab_size)。这样可允许在生成时只计算必要位置的 logits，避免为所有历史位置计算 logits，从而提高效率

        loss = None #置空
        if labels is not None: #label不为空
            shift_logits = logits[..., :-1, :].contiguous() #取 logits 中除了最后一个 token 之外的所有 token 的预测（即每个位置预测下一个 token）。contiguous() 确保张量在内存中是连续的，以便 view 操作
            shift_labels = labels[..., 1:].contiguous() #取 labels 中除了第一个 token 之外的所有 token（即每个位置的目标是下一个 token）。contiguous() 确保张量在内存中是连续的，以便 view 操作
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100) #将 logits 重塑为二维 (batch * (seq_len-1), vocab_size)，labels 重塑为一维，然后计算交叉熵损失。ignore_index=-100 表示忽略标签中值为 -100 的位置（通常用于填充 token 或无需计算损失的部分）。

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states) #CausalLMOutputWithPast 是 Transformers 库中定义的一个数据类，通常包含：loss：计算得到的损失（可选）。logits：预测的 logits。past_key_values：缓存（用于后续生成）。hidden_states：最后一层隐藏状态（用于调试或分析）
        output.aux_loss = aux_loss #将辅助损失作为额外属性添加到输出对象中，这样在训练时可以从输出中提取 aux_loss 并与主损失相加。
        return output #返回输出
