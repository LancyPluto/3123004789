import re
import jieba
from collections import Counter
import math
import sys
import os
def clean_text(text: str) -> str:
    """清理文本，去除非中文、英文字符"""
    if not isinstance(text, str):
        raise TypeError("输入必须是字符串")
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("清理后文本为空")
    return cleaned

def tokenize(text: str) -> list[str]:
    """中文分词或英文分词"""
    return list(jieba.cut(text))

def lcs_length(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度"""
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def cosine_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    """基于词频向量的余弦相似度"""
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)

    # 计算内积
    intersection = set(a_counts.keys()) & set(b_counts.keys())
    dot_product = sum(a_counts[x] * b_counts[x] for x in intersection)

    # 计算模长
    norm_a = math.sqrt(sum(v ** 2 for v in a_counts.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b_counts.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
def compute_duplicate_rate(orig: str, copy: str) -> float:
    """计算重复率（余弦相似度版本）"""
    orig_clean = clean_text(orig)
    copy_clean = clean_text(copy)

    orig_tokens = tokenize(orig_clean)
    copy_tokens = tokenize(copy_clean)

    if not orig_tokens:
        raise ZeroDivisionError("原文无有效词，无法计算重复率")

    sim = cosine_similarity(orig_tokens, copy_tokens)
    return round(sim * 100, 2)


def read_text_file(path: str) -> str:
    """读取文件内容"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python work.py <原文文件路径> <抄袭文件路径>")
        sys.exit(1)

    orig_path = sys.argv[1]
    copy_path = sys.argv[2]

    try:
        # 读取文件
        orig_text = read_text_file(orig_path)
        copy_text = read_text_file(copy_path)

        # 计算重复率
        rate = compute_duplicate_rate(orig_text, copy_text)

        # 输出结果
        print("=========== 查重结果 ===========")
        print(f"原文文件 : {orig_path}")
        print(f"抄袭文件 : {copy_path}")
        print(f"重复率   : {rate:.2f}%")
        print("================================")

    except Exception as e:
        print(f"[错误] {e}")

if __name__ == "__main__":
    # 如果只输入两个参数 -> 正常运行
    if len(sys.argv) == 3:
        main()
    else:
        # 调试模式：直接运行性能分析
        from pyinstrument import Profiler

        text1 = "我喜欢自然语言处理" * 1000
        text2 = "自然语言处理是人工智能的重要分支" * 1000

        profiler = Profiler()
        profiler.start()

        # 调用你要分析的函数
        rate = compute_duplicate_rate(text1, text2)

        profiler.stop()

        print("重复率:", rate)
        print(profiler.output_text(unicode=True, color=True))

        # 打开火焰图（浏览器中查看函数耗时分布）
        profiler.open_in_browser()

