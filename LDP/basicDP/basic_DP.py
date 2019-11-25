"""
@author:FZX
@file:basic_DP.py
@time:2019/11/25 16:21
"""


import numpy as np


def epsilon2probability(epsilon, n=2): # n=2是这里只有二元的，如果是多元的可以设置为n
    return np.e ** epsilon / (np.e ** epsilon + n - 1)


def discretization(value, lower, upper):
    """discretiza values
    :param value: value that needs to be discretized
    :param lower, the lower bound of discretized value
    :param upper: the upper bound of discretized value
    :return: the discretized value
    """
    if value > upper or value < lower:
        raise Exception("the range of value is not valid in Function @Func: discretization")

    p = (value - lower) / (upper - lower)
    rnd = np.random.random()  # 生成取值范围为[0,1)范围的随机浮点数
    return upper if rnd < p else lower #如果rnd小于p就输出upper，大于p就输出lower


def perturbation(value, perturbed_value, epsilon):
    """
    perturbation, (random response is a kind of perturbation) 扰动函数
    :param value: the original value
    :param perturbed_value: the perturbed value
    :param epsilon: privacy budget
    :return: dp version of perturbation
    """
    p = epsilon2probability(epsilon)
    rnd = np.random.random() # 生成取值范围为[0,1)范围的随机浮点数
    return value if rnd < p else perturbed_value # 以小于p的概率输出原始值，以1-p的概率输出扰动值


def random_response_basic(bit, epsilon):
    if bit not in [0, 1]:
        raise Exception("The input value is not in [0, 1] @Func: random_response.")
    return perturbation(value=bit, perturbed_value=1 - bit, epsilon=epsilon)# 以小于p的概率输出原始值，以1-p的概率输出翻转的扰动值


def random_response_pq(bits, probability_p, probability_q):
    """
    This is the generalized version of random response. When p+q=1, this mechanism turns to be the basic random response.这里如果p+q=1就是上面的方法
    See this paper: Locally Differentially Private Protocols for Frequency Estimation
    :param bits: the original data.
    :param probability_p: the probability of 1->1
    :param probability_q: the probability of 0->1
    :return: the perturbed bis
    """
    if not isinstance(bits, np.ndarray):
        raise Exception("the input type is not illegal @Func: random_response_pq.")

    index_one, index_zero = (bits == 1), (bits == 0)
    # flags is used to represent flip or not, 1 represents unchanged, 0 represents flipping.
    flags = np.zeros([bits.size], dtype=int)
    flags[index_one] = np.random.binomial(n=1, p=probability_p) #二项分布
    flags[index_zero] = np.random.binomial(n=1, p=1-probability_q)
    res = 1 - (bits + flags) % 2
    return res
"""

n：int型或者一个int型的数组，大于等于0，接受浮点数但是会被变成整数来使用。
p：float或者一组float的数组，大于等于0且小于等于1.
size：可选项，int或者int的元祖，表示的输出的大小，如果提供了size，例如(m,n,k)，那么会返回m*n*k个样本。如果size=None，也就是默认没
有的情况，当n和p都是一个数字的时候只会返回一个值，否则返回的是np.broadcast(n,p).size个样本.
return：一个数字或者一组数字
每个样本返回的是n次试验中事件A发生的次数。
"""

def random_response_pq_reverse(sum_of_bits, num_of_records, probability_p, probability_q):
    """
    解码器
    decoder for function @random_response_pq
    :param sum_of_bits:
    :param num_of_records:
    :param probability_p: the probability of 1->1
    :param probability_q: the probability of 0->1
    :return:
    """
    return (sum_of_bits - num_of_records * probability_q) / (probability_p - probability_q)
"""
这里原论文中的decoder公式实际上是
(sum_of_bits - num_of_records * (probability_q+0.5*f*probability_p-0.5*f*probability_q)) / ((1-f)*(probability_p - probability_q)
不过这里f为0，所以就是上面的公式
还有p和q和RAPPOR论文中是倒过来的
"""


def coin_flip(bits, epsilon):
    """
    the coin flip process for bit array, it is random response with length = len(bits).
    :param bits: the original data
    :param epsilon: privacy budget
    :return: the perturbed data
    example, bits = [1,1,0,0], flags = [0,1,0,1], res = [0,1,1,0]
    """
    flags = np.random.binomial(n=1, p=epsilon2probability(epsilon), size=len(bits))
    res = 1 - (bits + flags) % 2
    return res


def random_response_adjust(sum, N, epsilon):
    """
    对random response的结果进行校正
    :param sum: 收到数据中1的个数
    :param N: 总的数据个数
    :return: 实际中1的个数
    """
    p = epsilon2probability(epsilon)
    return (sum + p*N - N) / (2*p - 1)


if __name__ == '__main__':
    a = np.asarray([1, 1, 1, 0, 0, 1, 0])
    print(random_response_pq(bits=a, probability_p=1, probability_q=0.9))