import torch
import json
from scripts.my_train import parse_args, build_model, get_optimizer
from cs336_basics.my_data_utils import my_load_checkpoint
from tokenizers import Tokenizer


def preview_tensor(t: torch.Tensor, last_n: int = 5):
    """Print up to `last_n` values from last dim, with leading dims fixed to 0."""
    if t.numel() == 0:
        print("[empty tensor]", t.shape, t.dtype, t.device)
        return
    if t.dim() == 0:
        print("scalar:", t.item())
        return
    # 构造索引：前面维度都取 0，最后一维切片
    idx = (0,) * (t.dim() - 1) + (slice(None, last_n),)
    sliced = t[idx]
    to_print = sliced.detach().cpu()
    #print(f"shape={tuple(sliced.shape)} (from {tuple(t.shape)}), dtype={t.dtype}, device={t.device}")
    print(to_print)

def get_tokenizer():
    tokenizer = Tokenizer.from_file("hf_tokenizer/tinystories/tokenizer.json")
    return tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, prompt_text, max_new_tokens=50, temperature=1.0, top_k=50, device="cpu"):
    # 编码提示
    enc = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(enc.ids, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(max_new_tokens):
        logits = model(input_ids)  # 期望输出形状 [B, T, V]
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        # top-k 过滤
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            min_keep = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < min_keep, torch.full_like(next_logits, -float("inf")), next_logits)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args,dtype=torch.float32,device=device)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}",end=" ")
    #     preview_tensor(param)

    optimizer = get_optimizer(model, args)
    my_load_checkpoint("my_models/TinyStories_17M_autodl.pt", model, optimizer)
    print("Model loaded successfully")

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}",end=" ")
    #     preview_tensor(param)

    tok = get_tokenizer()
    # vocab = tok.get_vocab()  # dict: token -> id
    # print("vocab size:", len(vocab))
    # print("sample vocab items:", list(vocab.items())[:20])

    # model_json = json.loads(tok.to_str())
    # print("merges count:", len(model_json["model"]["merges"]))
    # print("first 20 merges:", model_json["model"]["merges"][:20])

    prompt = "Once upon a time"
    # prompt = ""
    generated = generate_text(
        model,
        tok,
        prompt_text=prompt,
        max_new_tokens=200,
        temperature=0.8,
        top_k=40,
        device=device,
    )
    print("\n--- Generated ---")
    print(generated)




if __name__ == "__main__":
    args = parse_args()
    main(args)