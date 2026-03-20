import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore') #忽略警告信息


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):#训练批次循环
    start_time = time.time() #记录本epoch开始的时间
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):## 遍历数据加载器，start_step 用于从断点续训时从特定 Step 开始
        input_ids = input_ids.to(args.device) #输入token ids迁移到指定设备
        labels = labels.to(args.device) #将标签迁移到指定设备
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)#计算全局总步数等等，按照余弦退火公式获取当前的动态学习率
        for param_group in optimizer.param_groups: #遍历优化器的参数组
            param_group['lr'] = lr #应用计算出的动态学习率

        with autocast_ctx:#混合精度下的前向传播
            res = model(input_ids, labels=labels) #调用模型前向输出
            loss = res.loss + res.aux_loss #计算总损失==预测损失(loss) + 负载均衡辅助损失(aux_loss），16精度
            loss = loss / args.accumulation_steps #误差平均化
        scaler.scale(loss).backward() #在反向传播前先对loss进行放大（防止计算出的梯度精度不足被截断为0），再进行反向传播生成梯度

        if (step + 1) % args.accumulation_steps == 0:#每当达到触发梯度累计更新的步数时
            scaler.unscale_(optimizer) #先缩放回原来的大小，在fp32精度下缩放回去
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)#再对梯度进行裁剪，防止梯度爆炸

            scaler.step(optimizer) #更新模型参数
            scaler.update() #根据本次训练梯度情况更新下一次的缩放因子

            optimizer.zero_grad(set_to_none=True)## 清空梯度缓冲区，set_to_none=True 能略微提升性能并节省显存

          #以下是日志打印、监控
        if step % args.log_interval == 0 or step == iters - 1: #到了打印的步数或该轮次最后一步
            spend_time = time.time() - start_time #花费的时间
            current_loss = loss.item() * args.accumulation_steps #还原真实的loss，即将之前除的步数还原回去
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0 #aux损失，若没有则置0
            current_logits_loss = current_loss - current_aux_loss #预测损失
            current_lr = optimizer.param_groups[-1]['lr']  #将学习率从优化器的参数组中读出
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60 #简单估算本epoch剩余的分钟数
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')#自定义日志打印到控制台或者文件中
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min}) #若开启了wandb,将数据同步到云端看板
           #模型保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():#到了保存的步数或该轮次最后一步
            model.eval() #切换到评估模式
            moe_suffix = '_moe' if lm_config.use_moe else '' #moe设置
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth' #构建保存路径
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model 
            raw_model = getattr(raw_model, '_orig_mod', raw_model) # 处理分布式训练或 torch.compile 带来的模型封装问题
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp) ## 将模型转为 FP16 并移至 CPU 后保存，极大压缩文件体积且不占显存
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')## 调用外部函数保存更完整的检查点（含优化器状态、Step等），用于后续断点续训
            model.train() #恢复训练模式
            del state_dict #及时删除引用，释放内存

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="D:/my_llm/minimind/dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode() #初始化分布式训练环境，返回当前进程在当前机器上的 GPU 编号
    if dist.is_initialized(): args.device = f"cuda:{local_rank}" #如果分布式环境启动成功，为当前进程指定对应的 CUDA 设备
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0)) ## 设置全局随机种子，确保实验可复现；加上 rank 值是为了防止不同进程获取完全相同的数据顺序（去相关性）
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)## 创建保存权重的目录，如果目录已存在则跳过
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))# 根据参数实例化模型配置对象（如隐藏层维度、层数、是否使用混合专家模型 MoE
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None ## 如果设置了从断点恢复（from_resume=1），则从指定目录加载 checkpoint 数据
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu" # 判断运行设备类型是 GPU 还是 CPU
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 ## 根据参数选择数据精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype) ## 设置自动混合精度（AMP）的上下文。如果是 CPU 则不操作；如果是 GPU 则开启 autocast 以加速训练
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():## 仅在主进程（Rank 0）中初始化日志工具，避免多个进程重复记录
        import swanlab as wandb #实际使用 swanlab
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None# 如果有断点信息，则强制恢复之前的 wandb 会话，保持图表连续
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device) #模型、分词器的设置，加载配置、模型参数权重和设备
    if args.use_compile == 1:# 如果开启了编译优化
        model = torch.compile(model) #通过计算图优化提升训练速度
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)# 实例化预训练数据集对象
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None ## 分布式采样器：确保在 DDP 模式下，不同 GPU 看到的是数据集中不同的子集
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16')) # 梯度缩放器：配合 float16 训练使用，防止梯度下溢
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)## 定义 AdamW 优化器
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:# 如果存在 checkpoint
        model.load_state_dict(ckp_data['model'])#则恢复模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])#恢复优化器状态
        scaler.load_state_dict(ckp_data['scaler'])#恢复缩放器状态
        start_epoch = ckp_data['epoch']#恢复训练轮数
        start_step = ckp_data.get('step', 0)#恢复训练步数
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}## 忽略不需要同步梯度的缓冲区（如 RoPE 位置编码的预计算常量）
        model = DistributedDataParallel(model, device_ids=[local_rank])# 将模型封装为分布式数据并行模型，并在指定的 GPU 上运行
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):#从初始轮次开始循环
        train_sampler and train_sampler.set_epoch(epoch)  #如果使用了分布式训练（train_sampler 不为 None），调用 set_epoch(epoch) 来确保每个进程在每个 epoch 获得不同的数据划分，避免不同 epoch 的数据分布相同
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist() #每个 epoch 使用不同的种子（42 + epoch），保证数据打乱的可复现性。这样即使中断后恢复，也能按相同的顺序继续。orch.randperm 生成一个随机排列的索引列表，用于后续的 batch 采样。这通常在没有使用 DistributedSampler 时，用于自定义采样
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0 #仅在第一个恢复的 epoch 且 start_step 大于 0 时，才需要跳过前 start_step 个 batch。否则 skip = 0。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip) #创建 SkipBatchSampler：这是一个自定义的采样器包装类，它会从底层的采样器（可能是 train_sampler 或打乱的索引列表）中跳过前 skip 个 batch，然后返回后续的 batch。这样 DataLoader 就会从正确的位置开始产出数据
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True) #构建 DataLoader：使用上面创建的 batch_sampler，设置工作进程数、内存锁定等参数。
        if skip > 0: #如果是恢复训练（需要跳过）
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始') #记录日志
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb) #并调用 train_epoch，传入的 total_steps 是当前 loader 的长度加上跳过的步数（因为 loader 实际只包含剩余的 batch，但总步数应该是整个 epoch 的步数），start_step 作为起始步数，用于内部计算全局步数或进度条
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)#不然调用train_epoch，起始步数为0
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()