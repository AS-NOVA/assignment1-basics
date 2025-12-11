import argparse



# 参数
def parse_args():
    parser = argparse.ArgumentParser(description="cs336 ass1 training script")
    parser.add_argument("--example_arg", type=int, default=42, help="An example argument")
    pass

    args = parser.parse_args()
    return args

# model 与 tokenizer







# Dataset 与 DataLoader




# optimizer 与 scheduler






# 全流程，training loop
def main(args):
    pass




# 入口
if __name__ == "__main__":
    args = parse_args()
