import argparse
import sys
import torch
import wandb
from cs336_basics.my_module import TransformerLanguageModel


# 参数
def parse_args():
    parser = argparse.ArgumentParser(description="cs336 ass1 training script")

    # 路径配置：模型，数据，存档
    parser.add_argument("--train_data", type=str,   default="data/tinystories_bin/train.bin",   help="训练数据路径")
    parser.add_argument("--test_data",  type=str,   default="data/tinystories_bin/val.bin",     help="验证数据路径")
    parser.add_argument("--output_dir", type=str,   default="my_models/TinyStories_17M",      help="模型保存目录")

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
    parser.add_argument("--lr_min",         type=float, default=3e-5,   help="最小学习率") # 设为最大学习率的十分之一
    parser.add_argument("--total_iters",    type=int,   default=5000,   help="总迭代次数") # 若序列长256，batch大小256，投入token数327680000，恰需5000步
    parser.add_argument("--warmup_iters",   type=int,   default=500,    help="学习率热身步数")    # 总步数的10%
    parser.add_argument("--beta1",          type=float, default=0.9,    help="AdamW 一阶矩估计的衰减率")
    parser.add_argument("--beta2",          type=float, default=0.999,  help="AdamW 二阶矩估计的衰减率")
    parser.add_argument("--eps",            type=float, default=1e-8,   help="AdamW 防止除零的平滑项")
    parser.add_argument("--weight_decay",   type=float, default=0.01,   help="AdamW 权重衰减系数")

    # 训练配置：种子，批量大小
    parser.add_argument("--seed",           type=int,   default=42,     help="随机种子")
    parser.add_argument("--batch_size",     type=int,   default=256,    help="批量大小")
    
    args = parser.parse_args()
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
def build_model(args):
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    return model

def get_tokenizer(args):
    pass





# Dataset 与 DataLoader




# optimizer 与 scheduler






# 全流程，training loop
def main(args):
    print_args(args)
    confirm_to_start_training()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.cuda.empty_cache()    # 从dlk抄的
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    print("训练过程占位文本")

# 入口
if __name__ == "__main__":
    args = parse_args()
    main(args)