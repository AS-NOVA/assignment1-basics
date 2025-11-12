from cs336_basics.pretokenization_example import *
import regex as re

# from IPython.display import clear_output, display

# uv run pytest tests/test_train_bpe.py

# 从文件路径读入字节序列，用给出的切分函数找到chunk的边界，分段读入
def path_to_chunks_bytes(p:str | os.PathLike, n_parallel: int) -> list[bytes] :
    res = []
    with open(p,"rb") as f:
        num_processes = n_parallel
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start) #.decode("utf-8", errors="ignore")
            res.append(chunk)
    return res

# 对于每段字符串，去掉特殊token并预分词，返回预分词列表
# 曾经接收bytes，现改为str，方便在tokenizer中复用，因为那里的习题要求接收str而非bytes
def pre_tokenization_for_chunk(text_chunk:str, special_tokens: list[str]|None = None, keep_special_tokens:bool = True) -> list[str]:
    """
    给出一段文本，将其进行预分词，返回预分词得到的pre-token列表。
    允许用户设置特殊token，它们将被单独优先分割，且不会与其他内容混在一起。

    Args:
        text_chunk (str): 输入文本，字符串形式。
        special_tokens (list[str] | None = None) : 特殊token列表，字符串形式，默认为空。
            特殊token将会优先被匹配，自成一个pre-token，而不会与其他内容混在一起。
        keep_special_tokens (bool = True) : 预分词结果中是否保留特殊token，默认保留。
            如果保留特殊token，就能够保证正确地将全部原文本进行合理的预分词。
            不保留特殊token的形式主要用于bpe训练过程，可以防止特殊token自身的内容干扰不断merge的过程。

    Returns:
        list[str] : 原文本预分词的结果，即一个字符串形式的pre-token列表。
    """
    # 如果没有special tokens，必须特判，否则导致re.split用空串分割所有东西，导致所有字符被拆开
    if not special_tokens:
        #print("无special tokens")
        splited_text = [text_chunk]
    else:
        # text_chunk = text_chunk_bytes.decode("utf-8")
        special_tokens.sort(reverse=True) # 降序排序，以保证：在需要保留特殊token的预分词过程中，如果一个特殊token中包含另一个，会优先匹配长的那个
        escaped_special_tokens = [re.escape(t) for t in special_tokens]
        escaped_special_tokens_in_one_str = "|".join(escaped_special_tokens)
        # print("所有特殊token组合起来得到的用于re.split切分的串：",escaped_special_tokens_in_one_str)
        if keep_special_tokens == True:
            escaped_special_tokens_in_one_str = "("+escaped_special_tokens_in_one_str+")"
        elif keep_special_tokens == False:
            pass
        # # print(escaped_special_tokens_in_one_str)
        splited_text = re.split(escaped_special_tokens_in_one_str,text_chunk)
        # print("re.split将原始文本在specialtoken处切断（可能保留或不保留切分点），得到的结果：",splited_text)

    pre_tokenization = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in splited_text:
        # ！！！注意 如果要保留special token，就还需要保护其所在的片段，免受预分词！！！
        if keep_special_tokens and special_tokens and doc in special_tokens:
            pre_tokenization.append(doc)
            continue
        it = re.finditer(PAT,doc)
        for match in it:
            pre_tokenization.append(match.group())
    return pre_tokenization


def get_all_pretoken_bytes_and_build_count_dict(pre_tokens:list[list[str]]) -> dict[bytes,int]:
    res = {}
    for trunks in pre_tokens:
        for tok in trunks:
            btok = tok.encode("utf-8")
            res[btok] = res.get(btok,0) + 1
    return res

type IntsCount = dict[tuple[int,...],int]
type IntPairInInts = dict[tuple[int,int],set[tuple[int,...]]]

def build_inverted_index(pre_token_ints_dict:IntsCount) -> IntPairInInts:
    inverted_index = {}
    for key in pre_token_ints_dict.keys():
        pairs = zip(key[:-1],key[1:])
        for pair in pairs:
            inverted_index.setdefault(pair,set()).add(key)
    return inverted_index

class BpeManager:
    data: IntsCount
    inv_index: IntPairInInts
    vocab_dict: dict[int,bytes]
    merge_list:list[tuple[bytes,bytes]]

    def __init__(self, pre_token_ints_dict:IntsCount, special_tokens:list[str]) -> None:
        self.data = pre_token_ints_dict
        self.inv_index = build_inverted_index(pre_token_ints_dict=pre_token_ints_dict)
        self.vocab_dict = {n: bytes([n]) for n in range(256)}
        for st in special_tokens:
            self.vocab_dict[len(self.vocab_dict)] = st.encode("utf-8")
        self.merge_list = []
    
    # 单个单词中数出现次数
    def get_int_pair_occur_in_ints(self,intpair:tuple[int,int], ints: tuple[int,...]) -> int:
        count = 0
        for i in range(len(ints) - 1):
            if ints[i] == intpair[0] and ints[i+1] == intpair[1] :
                count += 1
        return count

    # 多个单词中数出现次数
    def get_int_pair_occur_times(self,intpair:tuple[int,int]) -> int:
        sum = 0
        for pretokints in self.inv_index[intpair]:
            sum += self.get_int_pair_occur_in_ints(intpair,pretokints) * self.data[pretokints]
        return sum

    def get_max_token_pair_int(self) -> tuple[int,int]:
        maxpair = max(
            self.inv_index, 
            key = lambda k : (                          # 这里k代指self.inv_index的key中的每个元素，即一个整数对
                self.get_int_pair_occur_times(k),       # 先按其出现次数排
                self.vocab_dict[k[0]],                  # 再看第一位的bytes
                self.vocab_dict[k[1]]                   # 最后看第二位的bytes
            )
        )
        # print("---寻找出现最多的token对---\n",
        #     f"出现最多的token对是{maxpair}，即{self.vocab_dict[maxpair[0]]}与{self.vocab_dict[maxpair[1]]}\n",
        #     f"出现了{self.get_int_pair_occur_times(maxpair)}次\n",
        #     f"出现在这些pretoken中（前10个）：{list(self.inv_index[maxpair])[:10]}\n",
        #     "---寻找结束---\n")
        return maxpair

    def build_new_token(self,new_pair_id:tuple[int,int]) -> tuple[int,bytes,int,bytes,int,bytes]:
        # 在vocab_dict中加入新token nt = lt + rt
        # 在merge中加入新merge (lt,rt)
        li = new_pair_id[0]
        ri = new_pair_id[1]
        lt = self.vocab_dict[li]
        rt = self.vocab_dict[ri]
        ni = len(self.vocab_dict)
        nt = lt + rt
        self.vocab_dict[ni] = nt
        self.merge_list.append((lt,rt))
        return (li,lt,ri,rt,ni,nt)

    # 一次merge的全过程
    def merge(self, new_pair_id:tuple[int,int]) -> None:
        (li,lt,ri,rt,ni,nt) = self.build_new_token(new_pair_id=new_pair_id)
        # print("---开始合并---\n",
        #     "本次合并情况：\n",
        #     f"{li}--{lt}与{ri}--{rt}合并得到{nt}，编号为{ni}")
        
        # 从inv_index中找到所有包含 (li,ri) 的pretoken单词
        # assert (li,ri) in self.inv_index , "反向索引中未找到所合并的对"
        # print("他们出现在这些单词中（前10个）：",f"{list(self.inv_index[(li,ri)])[:10]}")

        # 合并过程
        # 两件事：维护反向索引，更新data
        # 注意：原始的data中的每种pretoken是必定互不相同的，并且永远不可能变得相同
        # 实际上，同一序列是有可能拆分为两种不同的token的，例如cat = c + at = ca + t，但是在bpe中永远不需要考虑这种事情

        # 取出(li,ri)对应的所有pretoken(...,xi,li,ri,yi,...)
        #     对于每个pretoken(...xi,li,ri,yi,...)：
        #         取出所有它包含的pair(zi,wi)（包括(li,ri)本身）
        #         对于每个pair(zi,wi)：
        #             找到它里面的那个pretoken(...,xi,li,ri,yi,...)
        #             （即使这个pair对应的其他单词里也有(li,ri)，也没事，因为总会遍历到的）
        #             删掉它！
        #         新建一个pretoken(...,xi,ni,yi,...)，加入语料库，次数与原本相同
        #         取出所有它包含的pair(zi,wi)
        #         对于每个pair(zi,wi)：
        #             加上(...,xi,ni,yi,...)
        #         从语料库中删除这个pretoken(...xi,li,ri,yi,...)
        #     删掉这个(li,ri)（删前确保里面所有的单词都已经被删掉了）

        while self.inv_index[(li,ri)] :
            nowpretok = self.inv_index[(li,ri)].pop()
            n = self.data[nowpretok]
            del self.data[nowpretok]
            for pair in zip(nowpretok[:-1],nowpretok[1:]):
                self.inv_index[pair].discard(nowpretok)
            newpretok = self.get_new_pretok_from_oldpretok_and_pair(nowpretok,(li,ri),ni=ni)
            self.data[newpretok] = n
            for pair in zip(newpretok[:-1],newpretok[1:]):
                self.inv_index.setdefault(pair,set()).add(newpretok)

        # 清理inv_index中空的pair
        for pair in list(self.inv_index.keys()):
            if self.inv_index[pair] == set():
                del self.inv_index[pair]

        # print("---合并结束---\n")

    def get_new_pretok_from_oldpretok_and_pair(self,oldpretok:tuple[int,...],pair:tuple[int,int],ni:int) -> tuple[int,...]:
        # 注意处理pair出现多次的情况！
        newpretok = ()
        i = 0
        while i < len(oldpretok):
            if i < len(oldpretok) - 1 and (oldpretok[i],oldpretok[i+1]) == pair:
                # 找到一个pair
                newpretok += (ni,)
                i += 2
            else:
                newpretok += (oldpretok[i],)
                i += 1
        return newpretok

    def quick_look(self):
        pass
        # print("------状态速览------")
        # print("当前的data：")
        # print(self.data)
        # print([bytes(pt) for pt in self.data])
        # print([[self.vocab_dict[i] for i in j] for j in self.data])
        # print("当前的token对位置索引inv_index：")
        # print(self.inv_index)
        # print([[(self.vocab_dict[k[0]],self.vocab_dict[k[1]]),v] for k,v in self.inv_index.items()])
        # clear_output(wait=True)
        # display(f"当前词表大小：{len(self.vocab_dict)}，最新token：{list(self.vocab_dict.items())[-1:]}")
        # print("当前的合并列表：")
        # print(self.merge_list)
        # print("------状态速览------")
        # print()



def main_bpe(
    pre_token_ints_dict: IntsCount,
    vocab_size: int,
    special_tokens: list[str]
    )->tuple[
        dict[int,bytes],                # vocab
        list[tuple[bytes,bytes]]        # merge
    ]:

    manager = BpeManager(pre_token_ints_dict=pre_token_ints_dict, special_tokens=special_tokens)

    while len(manager.vocab_dict) < vocab_size:
        
        manager.quick_look()
        if manager.inv_index == {}:
            # print("不再有任何相邻token对，合并过程提前结束")
            break
        maxpair = manager.get_max_token_pair_int()
        # print("获得的最大token对：",
        #       manager.vocab_dict[maxpair[0]],
        #       manager.vocab_dict[maxpair[1]]
        #       )
        manager.merge(maxpair)

    return (manager.vocab_dict, manager.merge_list)

def my_bpe(input_path:str | os.PathLike, vocab_size:int, special_tokens:list[str]):
    # 从路径读取文件，转为文件指针。注意读出来的是bytes类型
    chunks_bytes = path_to_chunks_bytes(p=input_path, n_parallel=4)
    # 对每段文本进行预分词
    pre_tokens = []
    for chunk in chunks_bytes:
        chunk_str = chunk.decode("utf-8")
        pretok = pre_tokenization_for_chunk(chunk_str, special_tokens,keep_special_tokens=False)
        pre_tokens.append(pretok)
    # 整合所有预分词结果并转为bytes类型的token，建立词频字典
    pre_tokens_bytes_dict = get_all_pretoken_bytes_and_build_count_dict(pre_tokens)
    # 转为整数组表示的token，建立词频字典
    pre_token_ints_dict = {tuple(pre_token):cnt for pre_token , cnt in pre_tokens_bytes_dict.items()}
    # 运行BPE算法
    (vocab,merge) = main_bpe(pre_token_ints_dict=pre_token_ints_dict, vocab_size=vocab_size, special_tokens=special_tokens)

    return (vocab,merge)















