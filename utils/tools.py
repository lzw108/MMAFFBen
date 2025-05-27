# Semeval

import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
import numpy as np


def get_scores_EIreg(text):
    match = re.search(r'\b0\.[0-9]+\b', text)

    if match:
        # 如果找到，返回第一个 0-1 之间的小数
        a = float(match.group())
        if a > 1:
            a = 1
        elif a < 0:
            a = 0
        return a
    else:
        pattern = r'-?\d+(?:\.\d+)?'  # r"[-+]?\d*\.\d+|\d+"  # 匹配浮点数的正则表达式模式
        floats = re.findall(pattern, text)
        try:
            a = [float(num) for num in floats][0]
        #         a = [float(num) for num in floats if 0<float(num)<1][0]
        except:
            a = 0.5
        if a > 1:
            a = 1
        elif a < 0:
            a = 0
        return float(a)


def extract_integers_EIoc(string):
    integers = re.findall(r'[-+]?\d+', string)
    try:
        a = [int(num) for num in integers][0]
    except:
        #         print(string)
        a = 0
    if a > 3:
        a = 3
    elif a < 0:
        a = 0
    return a


def extract_integers_Voc(string):
    integers = re.findall(r'[-+]?\d+', string)
    try:
        a = [int(num) for num in integers if -3 <= int(num) <= 3][0]
    except:
        #         print(string)
        a = 0
    if a > 3:
        a = 3
    elif a < -3:
        a = -3
    return a


def get_scores_Vreg(text):
    pattern = r'-?\d+(?:\.\d+)?'  # r"[-+]?\d*\.\d+|\d+"  # 匹配浮点数的正则表达式模式
    floats = re.findall(pattern, text)
    try:
        a = [float(num) for num in floats][0]
    except:
        a = 0
    if a > 1:
        a = 1
    elif a < -1:
        a = -1
    return a


def parse_labels_Ec(label_str):
    # 类别名称到编号映射
    """解析标签：
       - 优先解析类别名称，若存在则忽略数字编号
       - 若无类别名称，再考虑数字编号
       - 自动去重，防止同一个类别重复
    """
    CATEGORY_MAP = {
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
        "anticipation": 7,
        "love": 8,
        "optimism": 9,
        "pessimism": 10,
        "trust": 11,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    #     # 优先检查类别名称
    #     has_word_label = False
    #     for word in label_str.split():
    #         word = word.strip(",.")  # 去掉多余标点
    #         if word in CATEGORY_MAP:
    #             labels.add(CATEGORY_MAP[word])  # 添加类别编号
    #             has_word_label = True
    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def parse_labels_EWECT(label_str):
    """解析标签：
       - 优先解析类别名称，若存在则忽略数字编号
       - 若无类别名称，再考虑数字编号
       - 自动去重，防止同一个类别重复
    """
    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


# 定义函数，从文本中提取 -1 或 1
def extract_label_onlineshopping(text):
    if isinstance(text, str):
        if "-1" in text or "negative" in text.lower():
            return -1
        elif "1" in text or "positive" in text.lower():
            return 1
    #     print("error, or no answer","***text:")
    return 9  # 处理异常情况


def extract_label_MMS(text):
    if isinstance(text, str):
        if "-1" in text or "negative" in text.lower():
            return -1
        elif "0" in text or "neutral" in text.lower():
            return 0
        elif "1" in text or "positive" in text.lower():
            return 1
    #     print("error, or no answer","***text:")
    return 9  # 处理异常情况


def parse_labels_XED(label_str):
    """解析标签：
       - 优先解析类别名称，若存在则忽略数字编号
       - 若无类别名称，再考虑数字编号
       - 自动去重，防止同一个类别重复
    """
    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
        "anticipation": 7,
        "trust": 8,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from scipy.stats import pearsonr


def parse_labels_EMOTIC_emo(label_str):
    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "aversion": 6,
        "excitement": 7,
        "peace": 8,
        "affection": 9,
        "annoyance": 10,
        "anticipation": 11,
        "confidence": 12,
        "disapproval": 13,
        "disconnection": 14,
        "disquietment": 15,
        "doubt/confusion": 16,
        "embarrassment": 17,
        "engagement": 18,
        "esteem": 19,
        "fatigue": 20,
        "pain": 21,
        "pleasure": 22,
        "sensitivity": 23,
        "suffering": 24,
        "sympathy": 25,
        "yearning": 26,
    }
    num_classes = len(CATEGORY_MAP)

    """解析标签：
       - 适用于标准多标签分类（包含 0-11）
       - 允许解析类别名称和数字编号
       - 自动去重，防止同一个类别重复
    """
    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def extract_values_EMOTIC_vad(text):
    valence = re.search(r'valence\s*[:\-]?\s*([\-]?\d*\.\d+|\d+)', text)
    arousal = re.search(r'arousal\s*[:\-]?\s*([\-]?\d*\.\d+|\d+)', text)
    dominance = re.search(r'dominance\s*[:\-]?\s*([\-]?\d*\.\d+|\d+)', text)

    valence_value = float(valence.group(1)) if valence else 0.0
    arousal_value = float(arousal.group(1)) if arousal else 0.0
    dominance_value = float(dominance.group(1)) if dominance else 0.0

    return valence_value, arousal_value, dominance_value


# 计算皮尔逊相关系数
def calculate_pearson_correlation(col1, col2):
    return pearsonr(col1, col2)[0]


def parse_labels_FER2013(label_str):
    """解析标签：
       - 适用于标准多标签分类（包含 0-11）
       - 允许解析类别名称和数字编号
       - 自动去重，防止同一个类别重复
    """

    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def parse_labels_CFAPS_emo(label_str):
    """解析标签：
       - 适用于标准多标签分类（包含 0-11）
       - 允许解析类别名称和数字编号
       - 自动去重，防止同一个类别重复
    """

    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def get_scores_CFAPS_emoint(text):
    match = re.search(r'\b0\.[0-9]+\b', text)

    if match:
        # 如果找到，返回第一个 0-1 之间的小数
        a = float(match.group())
        if a > 1:
            a = 1
        elif a < 0:
            a = 0
        return a
    else:
        pattern = r'-?\d+(?:\.\d+)?'  # r"[-+]?\d*\.\d+|\d+"  # 匹配浮点数的正则表达式模式
        floats = re.findall(pattern, text)
        try:
            a = [float(num) for num in floats][0]
        #         a = [float(num) for num in floats if 0<float(num)<1][0]
        except:
            a = 0.5
        if a > 1:
            a = 1
        elif a < 0:
            a = 0
        return float(a)

# video
import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def parse_labels_SAMSEMO(label_str):
    """解析标签：
       - 适用于标准多标签分类（包含 0-11）
       - 允许解析类别名称和数字编号
       - 自动去重，防止同一个类别重复
    """

    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
        "other emotions": 99,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def parse_labels_MELD_emo(label_str):
    """解析标签：
       - 适用于标准多标签分类（包含 0-11）
       - 允许解析类别名称和数字编号
       - 自动去重，防止同一个类别重复
    """

    # 类别名称到编号映射
    CATEGORY_MAP = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
    }
    num_classes = len(CATEGORY_MAP)

    if not isinstance(label_str, str):
        return [0] * num_classes  # 处理缺失值

    # 统一处理分隔符，去除 "and"、"or"、"with"
    label_str = label_str.lower().replace("and", ",").replace("or", ",").replace("with", ",")

    labels = set()  # 使用集合防止重复

    # 优先检查类别名称
    has_word_label = False
    label_str_ = re.sub(r'[^\w\s,]', '', label_str)
    for word in label_str_.split():
        word = word.strip(",.")  # 去掉多余标点
        if word in CATEGORY_MAP:
            labels.add(CATEGORY_MAP[word])  # 添加类别编号
            has_word_label = True

    # 如果存在类别名称，忽略数字编号
    if not has_word_label:
        # 没有类别名称的情况下，才解析数字编号
        for num in re.findall(r'\b(\d+)\b', label_str):
            labels.add(int(num))  # 转换为整数

    # 生成二进制标签向量
    return [1 if i in labels else 0 for i in range(num_classes)]


def extract_label_MELD_sen(text):
    if isinstance(text, str):
        if "-1" in text or "negative" in text.lower():
            return -1
        elif "0" in text or "neutral" in text.lower():
            return 0
        elif "1" in text or "positive" in text.lower():
            return 1
    #     print("error, or no answer","***text:")
    return 9  # 处理异常情况


def extract_label_CHSIMS_pol(text):
    if isinstance(text, str):
        if "-1" in text or "negative" in text.lower():
            return -1
        elif "0" in text or "neutral" in text.lower():
            return 0
        elif "1" in text or "positive" in text.lower():
            return 1
    #     print("error, or no answer","***text:")
    return 9  # 处理异常情况


def get_scores_CHSIMS_strength(text):
    pattern = r'-?\d+(?:\.\d+)?'  # r"[-+]?\d*\.\d+|\d+"  # 匹配浮点数的正则表达式模式
    floats = re.findall(pattern, text)
    try:
        a = [float(num) for num in floats][0]
    except:
        a = 0
    if a > 1:
        a = 1
    elif a < -1:
        a = -1
    return a


def get_ACCandF1(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
#     accuracy = accuracy_score(y_true, y_pred)
    accuracy = jaccard_score(y_true,y_pred, average='weighted')

    return accuracy, micro_f1, macro_f1

