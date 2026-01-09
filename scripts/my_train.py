import argparse
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
from cs336_basics.my_module import TransformerLanguageModel, cross_entropy
from cs336_basics.my_optimizer import MyAdamW, cosine_annealing_with_warm_up, gradient_clipping
from cs336_basics.my_data_utils import my_get_batch, my_save_checkpoint
#from cs336_basics.my_


# 参数
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="cs336 ass1 training script")

    # 参数确认
    parser.add_argument('--no_confirm', action='store_false',default=True,dest="confirm", help='跳过参数确认，直接开始运行')

    # 路径配置：模型，数据，存档
    parser.add_argument("--train_data", type=str,   default="data/tinystories_bin/train.bin",   help="训练数据路径")
    parser.add_argument("--test_data",  type=str,   default="data/tinystories_bin/val.bin",     help="验证数据路径")
    parser.add_argument("--output_dir", type=str,   default="my_models/TinyStories_17M.pt",      help="模型保存目录")
    parser.add_argument("--not_save_model", action="store_false",default=True,dest="save_model", help="是否不保存模型")

    # Wandb 记录
    parser.add_argument("--wandb_project",      type=str, default="TinyStories_17M",    help="Wandb 项目名")
    parser.add_argument("--wandb_run_name",     type=str, default=None,                 help="Wandb Run 名")

    # 分词器配置
    parser.add_argument("--vocab_size",         type=int,   default=10000,  help="词汇表大小")

    # 模型配置：上下文，层数，头数，隐藏空间维数，位置编码
    parser.add_argument("--context_length", type=int,   default=256,    help="上下文长度")
    parser.add_argument("--num_layers",     type=int,   default=4,      help="层数")
    parser.add_argument("--num_heads",      type=int,   default=16,     help="注意力头数")
    parser.add_argument("--rope_theta",     type=float, default=10000,  help="RoPE 角度参数")
    parser.add_argument("--d_model",        type=int,   default=512,    help="模型维度")
    parser.add_argument("--d_ff",           type=int,   default=1344,   help="前馈网络维度")

    # 优化配置：AdamW参数，学习率调度参数，总步数
    parser.add_argument("--lr_max",         type=float, default=3e-4,   help="最大学习率")
    parser.add_argument("--lr_min_rate",    type=float, default=0.1,    help="最小学习率与最大学习率之比")
    parser.add_argument("--total_iters",    type=int,   default=10000,  help="总迭代次数")
    # 若序列长256，batch大小256，投入token数327680000，恰需5000步

    # 训练中评估配置：评估间隔，单次评估采样次数
    parser.add_argument("--eval_interval",  type=int,   default=500,    help="评估间隔步数")
    parser.add_argument("--eval_iters",     type=int,   default=100,    help="评估步数")

    parser.add_argument("--warmup_time",    type=float, default=0.1,    help="学习率热身步骤占比")
    parser.add_argument("--beta1",          type=float, default=0.9,    help="AdamW 一阶矩估计的衰减率")
    parser.add_argument("--beta2",          type=float, default=0.999,  help="AdamW 二阶矩估计的衰减率")
    parser.add_argument("--eps",            type=float, default=1e-8,   help="AdamW 防止除零的平滑项")
    parser.add_argument("--weight_decay",   type=float, default=0.01,   help="AdamW 权重衰减系数")

    # 训练配置：种子，批量大小
    parser.add_argument("--seed",           type=int,   default=42,     help="随机种子")
    parser.add_argument("--batch_size",     type=int,   default=128,     help="批量大小")
    
    args = parser.parse_args(argv)
    return args

def print_args(args):
    print("============== 训练参数配置 ==============")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=========================================")

def confirm_to_start_training():
    while True:
        response = input("是否开始训练？(y/n): ").strip().lower()
        if response == 'y':
            print("训练已开始...")
            break
        elif response == 'n':
            print("训练已取消。")
            sys.exit(0)
        else:
            print("输入无效，请输入 y 或 n")
    return

def set_seed(seed):
    """把random、np.random、torch的随机函数的种子都设置成指定的同一个"""
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# model 与 tokenizer
def build_model(args,dtype,device):
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        dtype=dtype,
        device=device,
    )
    return model

def get_tokenizer(args):
    pass





# Dataset 与 DataLoader




# optimizer 与 scheduler

def get_optimizer(model, args):
    optimizer = MyAdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    return optimizer

def get_scheduled_lr(it,args):
    return cosine_annealing_with_warm_up(
        it=it,
        max_learning_rate=args.lr_max,
        min_learning_rate=args.lr_max * args.lr_min_rate,
        warmup_iters=args.total_iters * args.warmup_time,
        cosine_cycle_iters=args.total_iters,
    )

# 评估

@torch.no_grad()
def evaluate(model:TransformerLanguageModel, data, args, device):
    """
    Estimates the loss over a number of batches.
    """
    model.eval()
    losses = []
    entropies = []
    for k in tqdm(range(args.eval_iters), desc="Evaluating", leave=False):
        x, y = my_get_batch(data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
        # entropies.append(compute_entropy_chunked(logits).mean().item())
    model.train()
    mean_loss = np.mean(losses)
    return {
        'val/loss': mean_loss,
        # 'val/ppl': np.exp(mean_loss),
        # 'val/entropy': np.mean(entropies)
    }

# 全流程，training loop
def main(args):


    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.cuda.empty_cache()    # 从dlk抄的
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    print("正在加载数据...")
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.test_data, dtype=np.uint16, mode='r')
    print(f"训练数据token数: {len(train_data)}, 验证数据token数: {len(val_data)}")

    model = build_model(args,dtype=torch.float32,device=device)
    print("模型构建完成。")
    # model = torch.compile(model)
    # for name, p in model.named_parameters():
    #    print(name, p.shape, p.requires_grad)

    optimizer = get_optimizer(model, args)
    print("优化器构建完成。")

    print("开始训练...")
    for it in tqdm(range(args.total_iters)):
    # for it in tqdm(range(100)):
        lr = get_scheduled_lr(it, args)
        for param_group in optimizer.param_groups:  # 其实只有一个参数组
            param_group['lr'] = lr

        input, labels = my_get_batch(train_data, args.batch_size, args.context_length, device)

        # 前向传播
        logits = model(input)
        loss = cross_entropy(logits, labels)
        wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=it)
        # 还可以（每过一段时间）监视分布的熵，梯度的范数，等等

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        if (it + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(model, val_data, args, device)
            wandb.log(eval_metrics, step=it)
            print(f"Step {it}: " + ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()]))

        # 优化一步
        clipped_grad = gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

    if args.save_model:
        my_save_checkpoint(model, optimizer, it, args.output_dir)


# 入口
if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    if args.confirm:
        confirm_to_start_training()
    main(args)