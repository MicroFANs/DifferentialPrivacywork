"""
@author:FZX
@file:RPbasic.py
@time:2019/12/10 10:29
"""
import numpy as np


def gen_probability(epsilon, n=2):
    """
    根据epsilon生成概率p，默认是二元的n=2,以p的概率不变，1-p的概率翻转
    :param epsilon: 预算
    :param n: n元的
    :return: 概率值(e^eps)/(e^eps+1)
    """
    return np.e ** epsilon / (np.e ** epsilon + n - 1)




def gen_f(epsilon):
    """
    根据epsilon生成PRR中的参数f，PRR中是以1-0.5f概率保持不变，以0.5f翻转
    :param epsilon:
    :return:
    """
    f = 2 / (1 + np.e ** epsilon)
    return f


def perturbation(bit, p):
    """
    扰动，以概率p保持不变，以概率1-p翻转
    :param value: 输入
    :param probability:概率p
    :return: 扰动bit
    """
    rnd = np.random.random()
    perturbed_bit = 1 - bit
    return bit if rnd < p else perturbed_bit


def random_response_basic(bit, epsilon):
    """
    最基本的random_response
    :param bit:
    :param epsilon:
    :return:
    """
    if bit not in [0, 1]:
        raise Exception("The input value is not in [0, 1] @Func: random_response.")
    p = gen_probability(epsilon)
    return perturbation(bit, p)


def PRR(bit, f):
    """
    RAPPOR中针对单个比特位的PRR，实际上就是p=1-0.5f的基本RR
    :param bit:
    :param f: 概率f
    :return: 扰动后的bit
    """
    p = 1 - 0.5 * f
    return perturbation(bit, p)


def PRR_bits(bits, f):
    """
    针对比特串（numpy数组）的PRR
    :param bits:
    :param f:
    :return:

    flag是跟原数据一样长的数组，每一位独立随机数以p的概率产生1，以1-p的概率
    产生0，其中1表示该位不变，0表示该位翻转
    """
    p = 1 - 0.5 * f
    flag = np.random.binomial(n=1, p=p, size=len(bits))
    res = 1 - (bits + flag) % 2
    return res


def IRR(bit, q, p):
    """
    针对单个比特位的IRR过程
    :param bit:
    :param q: 1->1的概率
    :param p: 0->1的概率
    :return:
    """
    rnd = np.random.random()
    if bit == 1:
        S = bit if rnd < q else 1 - bit
    else:
        S = 1 if rnd < p else bit
    return S


def IRR_bits(bits, q, p):
    """
    针对比特串（numpy数组）的IRR过程
    :param bits: the original data.
    :param q: the probability of 1->1
    :param p: the probability of 0->1
    :return: the perturbed bits

    flag用来表示该比特位是否反转, 1表示不变, 0表示翻转
    example, bits = [1,1,0,0], flags = [0,1,0,1], res = [0,1,1,0]
    """
    # # x=np.array([1,0,1,0,1,0,1,0,1])
    # x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    # index_one, index_zero = (x == 1), (x == 0)
    # print('1:', index_one)
    # print('0:', index_zero)
    # flg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(flg[index_one], flg[index_zero])
    # len_1 = len(flg[index_one])
    # len_0 = len(flg[index_zero])
    # flg[index_one] = np.array([21, 21, 21, 21, 21])
    # flg[index_zero] = np.array([20, 20, 20, 20])
    # print(flg)

    if not isinstance(bits, np.ndarray):
        raise Exception("the input type is not illegal @Func: IRR_bits.")
    index_1, index_0 = (bits == 1), (bits == 0)
    flag = np.zeros([bits.size], dtype=int)
    len_1 = len(flag[index_1])
    len_0 = len(flag[index_0])
    # np.random.binomial(1,q,len_1)是生成长度为len_1的np数组，其中每一位是独立随机产生的0或1，产生1的概率为q
    flag[index_1] = np.random.binomial(1, q, len_1)  #
    flag[index_0] = (1 - np.random.binomial(1, p, len_0))
    print(flag)

    res = 1 - (bits + flag) % 2
    return res


def decoder_RR(sum, N, p):
    """
    以p的概率不变，以1-p的概率反转的解码
    :param sum:可以是单个值，也可以是numpy数组
    :param N:用户总数
    :param p:反转概率，在Basic RAPPOR中该p是1-0.5f
    :return:
    """
    return (sum + N * p - N) / (2 * p - 1)


def decoder_PRR(sum, N, f):
    """
    针对Onetime RAPPOR的解码，其中只包含了一次PRR
    :param sum:
    :param N:
    :param f:
    :return:
    """
    p = 1 - 0.5 * f
    return decoder_RR(sum, N, p)


def decoder_RAPPOR(sum, N, f, p, q):
    """
    RAPPOR论文中的计算每一位估计值的公式，包含了PRR和IRR两个过程
    :param sum: 每一位上1的和
    :param N: 总数
    :param f: PRR的参数，以1-0.5f的概率不变，以0.5f概率翻转
    :param p: 1->1的概率
    :param q: 0->1的概率
    :return:
    """
    est = (sum - (p + 0.5 * f * q - 0.5 * f * p) * N) / ((1 - f) * (q - p))
    return est


if __name__ == '__main__':
    x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    b = IRR_bits(x, q=0.75, p=0.5)
