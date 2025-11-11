from cs336_basics.mybpe import pre_tokenization_for_chunk
from typing import Iterable, Iterator

# uv run pytest tests/test_tokenizer.py

class MyTokenizer:
    """
    自制的bpe tokenizer类。指定vocab和merges以及一些特殊token后，能够对任意给定的文本进行编码，或对给定的id序列进行解码。
    Attributes:
        vocab (dict[int,bytes]) : 词汇表，按id查找token字节序列
        _reverse_vocab (dict[bytes,int]) : 反词汇表，按token字节序列查找id
        merges (list[tuple[bytes,bytes]]) : 训练过程中生成的merge序列
        _merges_rank (dict[tuple[bytes,bytes],int]) : merge序号表，便于O(1)查找到任何merge的序号，判断其顺序
        special_tokens (list[str]| None): 额外提供的特殊token，将来使用时必须将这些特殊token视为独立单位，而不能分割
    """
    def __init__(self,
                 vocab:dict[int,bytes],
                 merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]| None = None
                 )-> None:
        """
        接收vocab、merges和特殊token序列以进行分词器的初始化。
        初始化过程中会建立vocab的反表，并词典化merges，便于按输入内容进行encode。
        """
        self.vocab = vocab
        self._get_reverse_vocab()
        self.merges = merges
        self._get_merges_rank()
        self.special_tokens = special_tokens

    def _get_reverse_vocab(self) -> dict[bytes,int]:
        """
        返回self.vocab的逆表，即根据token字节序列查询其id。
        如果没计算过逆表，会重新计算并存入self.reverse_vocab。否则直接返回。
        Returns:
            dict[bytes,int]: 当前vocab的逆表，可以根据token字节序列查询其id。
        """
        # #print(self.vocab.items())
        if not hasattr(self,"reversed_vocab"):
            self._reverse_vocab = {tok:id for id, tok in self.vocab.items()}
        return self._reverse_vocab

    def _get_merges_rank(self) -> dict[tuple[bytes,bytes],int]:
        """
        返回self.merges转为的根据merge内容查询序号的列表，以便进行O(1)的查找。
        如果没有创建过哈希，会将其存入self._merges_rank。否则直接返回。
        Returns:
            dict[tuple[bytes,bytes],int]: 根据merge内容查找序号的列表。
        """
        if not hasattr(self,"_merges_rank"):
            self._merges_rank = dict([(merge, i) for (i, merge) in enumerate(self.merges)])
        return self._merges_rank

    @classmethod
    def from_files(cls,
                   vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens:list[str]|None = None
                   )->None:
        raise NotImplementedError

    def encode(self,text:str) -> list[int]:
        """
        基于存储的vocab和merge序列，输入文本字符串，将其预分词，按utf-8编码解读并进行bpe分词，最后返回token id序列。
        Arguments:
            text (str): 字符串形式的输入文本
        Returns:
            list[int]: 一个token id列表，即对输入文本的预分词结果
        """
        res = []
        pretoks = pre_tokenization_for_chunk(text,self.special_tokens,keep_special_tokens=True)
        #print("正在编码：",pretoks)
        pretoks_bytes = [pretok.encode("utf-8") for pretok in pretoks]
        for pretok_bytes in pretoks_bytes:
            # #print(pretok_bytes)
            ints = self._encode_for_bytes(pretok_bytes)
            # #print(ints)
            res += ints
        #print("编码结果：",res)
        return res



    def _encode_for_bytes(self,input:bytes) -> list[int]:
        """
        对于给定的字节序列，进行bpe分词，即不断进行merge过程，最终返回得到的token id序列。
        注意：这理应对任意字节序列都能运行，因为vocab中理应包含了所有单字节。即使一次merge也无法进行，也应该返回所有单字节本身的id。
        另外，对于special token，它们无法通过merge合并出来，所以需要直接用vocab进行查询。
        Arguments:
            input (bytes): 任意的字节序列
        Returns:
            out (list[int]): bpe分词得到的token id列表。
        """
        # 先找出特殊token
        # ！！！！注意：这里认为特殊token一定在vocab里，一定有一个能查到的序号！！！！
        if self.special_tokens != None and input.decode("utf-8") in self.special_tokens:
            if input not in self._get_reverse_vocab():
                raise KeyError(f"{input}不在词汇表中！")
            #print(f"{input}是一个特殊token，编号为{self._get_reverse_vocab()[input]}")
            return [self._get_reverse_vocab()[input]]
        else:
            # TODO:编写不断进行merge的过程。
            bytes_list = [bytes([b]) for b in input]
            #print(f"正在编码:{bytes_list}")
            while(len(bytes_list)>1):
                # step 1: 找到所有merge中序号最小的。如果没有，说明不能继续merge，应该直接将当前的所有token转为数字
                pairs_iter = zip(bytes_list[:-1],bytes_list[1:])
                #print(f"当前存在的token对：{[pair for pair in zip(bytes_list[:-1],bytes_list[1:])]}")

                valid_pairs = [pair for pair in pairs_iter if pair in self._get_merges_rank()]
                #print(f"当前可用于合并的token对：{valid_pairs}")
                min_pair = min(valid_pairs, 
                               key=lambda pair: self._merges_rank.get(pair, float("inf")), # 其实已经滤除了不在merges里的，所以并不会真的有inf出现
                               default=None)
                if min_pair == None:
                    #print(f"找不到可用的合并，停止合并并直接编码")
                    res = self._get_id_for_bytes_list(bytes_list)
                    return res
                #print(f"序号最小的merge：{min_pair}")
                # step 2: 如果找到了最小的对，就用它对bytes_list进行合并
                new_tok = min_pair[0]+min_pair[1]
                new_list = []
                i=0
                while i <= len(bytes_list)-1:
                    if i < len(bytes_list)-1 and bytes_list[i] == min_pair[0] and bytes_list[i+1] == min_pair[1]:
                        new_list.append(new_tok)
                        i += 2
                    else:
                        new_list.append(bytes_list[i])
                        i += 1
                
                # step 3: 合并后的新列表拿去继续循环
                #print(f"经过一轮merge，当前结果：{new_list}")
                bytes_list = new_list



            # 退出循环只有一种可能，就是bytes_list长度为1了
            #print(f"merge结束后的结果：{bytes_list}")
            res = self._get_id_for_bytes_list(bytes_list)
            #print(f"对应到词汇表：{res}")
            return res
    
    def _get_id_for_bytes_list(self,bytes_list:list[bytes])->list[int]:
        """
        把给定的字节形式的token列表翻译为token id列表。
        Arguments:
            bytes_list(list[bytes]): 输入的字节形式的token列表。使用时应当保证输入的每个字节串都是已知的token。
        Returns:
            out(list[int]): 将每个bytes翻译为token id的结果列表。
        """
        res = []
        for b in bytes_list:
            if b not in self._reverse_vocab:
                raise KeyError(f"{b}不在词汇表中")
            res.append(self._reverse_vocab[b])
        return res


    def encode_iterable(self,
                        iterable: Iterable[str]
                        )->Iterator[int]:
        raise NotImplementedError

    def decode(self,ids:list[int]) -> str:
        raise NotImplementedError