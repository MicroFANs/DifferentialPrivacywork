"""
@author:FZX
@file:SHbasic.py
@time:2019/11/25 20:46
"""
import numpy as np
from random import choice

def p_values(epsilon,n=2):
    """
    计算概率p的值,在RAPPOR中的敏感度需要从epsilon来体现，如果敏感度为h,则预算应该为(epsilon/h)
    :param epsilon: 隐私预算
    :param n: 默认n=2是二元的，也可以是多元的
    :return:概率p的值
    """
    p=np.e**epsilon/(np.e**epsilon+n-1)
    return p



def gen_m_bit_string(bits):
    """
    生成长度为m的字符串，元素为{-1/sqrt(m),1/sqrt(m)}
    :param bits:输入string，格式是np数组
    :return:输出
    """
    m=len(bits)
    down=1/np.sqrt(m)
    bits[bits==0]=-1*down
    bits[bits==1]=down
    return bits



def get_c_epsilon(epsilon):
    """
    计算c_epsilon
    :param epsilon:隐私预算
    :return:c_epsilon
    """
    c_epsilon=(np.e**epsilon+1)/(np.e**epsilon-1)
    return c_epsilon



def Basic_Randomizer(input,epsilon):
    """
    SHist论文中 Algorithm 1
    :param input: 长度为m的np数组，元素为{0，1}
    :param epsilon: 隐私预算
    :return: 向量z
    """
    # m-bit string x
    m=len(input)
    #x=gen_m_bit_string(input)  #输入input的形式已经是{-1/sqrt(m),1/sqrt(m)}的形式了，就无需转换了
    x=input

    # 随机选择一个序号j，j在区间[0,m)中
    j=choice(range(m))
    #print('j=',j+1) # 这里数组是从0开始的，但是显示的j应该是j+1

    # 计算c_epsilon
    c_eps=get_c_epsilon(epsilon)

    # 扰动
    p=p_values(epsilon) # 扰动概率p
    if np.all(x==0): # 如果字符串x为全0
        z_j=choice([-1*c_eps*np.sqrt(m),c_eps*np.sqrt(m)]) # 从二者任选一个
    else: # x不为全0，则以概率p变换
        rand=np.random.random() # 生成取值范围为[0,1)范围的随机浮点数
        if rand<p:z_j=c_eps*m*x[j]
        else:z_j=-1*c_eps*m*x[j]

    # 输出
    z=np.zeros(m)
    z[j]=z_j
    #print(z)
    return z



def genProj(m,d):
    """
    创建随机投影矩阵的规则，矩阵内的每个元素都是独立的、均匀的从{-1/sqrt(m),1/sqrt(m)}中选择
    :param m:
    :param d:
    :return:
    """
    random_projection_matrix=np.random.random((m,d))
    random_projection_matrix[random_projection_matrix<0.5]=-1/np.sqrt(m)
    random_projection_matrix[random_projection_matrix>=0.5] = 1 / np.sqrt(m)
    return random_projection_matrix



# def random_projection(d,n,epsilon,beta=0.00001):
# #     """
# #     根据参数d和置信度beta来确定投影矩阵
# #     :param d: 取值集合维度
# #     :param n: 所有用户的数量
# #     :param epsilon: 隐私预算
# #     :param beta: 置信度beta>0, 1-beta要很大，所以beta要小
# #     :return: 创建好的投影矩阵，以及m
# #     """
# #     r = np.sqrt((np.log(2 * d) / beta) / (n * epsilon ** 2))
# #     m = (np.log(d + 1) * np.log(2 / beta)) / (r ** 2)
# #     #print('m=',m)
# #     m=int(m)
# #     #print('m_int=',m)
# #     random_proj=genProj(m,d)
# #     return random_proj,m

def comput_parameter(d, n, epsilon, beta=0.05):
    """
    根据d和n计算r和m参数的值以便生成投影矩阵
    :param d: 用户数据维度d
    :param n: 用户数量
    :param epsilon: 隐私预算
    :param beta: 置信度，置信度beta>0, 1-beta要很大，所以beta要小
    :return: 参数r和参数m
    """
    r = np.sqrt((np.log(2 * d/ beta)) / ((epsilon ** 2)*n))
    m = (np.log(d + 1) * np.log(2 / beta)) / (r ** 2)
    m=int(m)
    return r,m



def Frequency_Oracle_Protocol(input,epsilon,random_proj):
    """
    SHist论文中 Algorithm 2，被拆成三部分，这里是客户端计算z_i的部分
    :param input:
    :param epsilon:
    :param random_proj:
    :return:z向量
    """
    x=np.matmul(random_proj,input)
    z=Basic_Randomizer(x,epsilon)
    return z



def Frequency_Estimator_Based_on_FO(random_proj,z_means,e_v):
    """
    SHist论文中Algorithm 3，计算v的频率估计值
    :param random_proj: 随机投影矩阵
    :param z_means: 收集的z均值，是一个向量
    :param e_v:需要计算估计值的编号所对应的onehot码，也就是基向量
    :return:f_v的频率估计
    """
    temp=np.matmul(random_proj,e_v)
    f_v_estimate=np.dot(temp,z_means)
    return f_v_estimate



# 编码 将v的值由{1,2...}映射成{-1/sqrt(m),1/sqrt(m)}^m
def encoder(input):
    codeword=input
    return codeword



#解码 将v的编码由{-1/sqrt(m),1/sqrt(m)}^m映射成{1,2...}
def decoder(codeword):
    origin=codeword
    return origin


def Succinct_Histogram_Protocol_user(input_v,epsilon,beta):
    """
    SHist论文中Algorithm4的用户端部分，该问题中用户都持有一个相同的项v*，对用户i的输入进行扰动，计算z_i
    :param input_v: 每个用户的输入
    :param epsilon: 隐私预算
    :param beta: 置信度
    :return:每个用户i的z_i
    """
    if input_v==0:x=np.zeros()
    else:x=encoder(input_v)
    z=Basic_Randomizer(x,epsilon)
    return z


def Succinct_Histogram_Protocol_server(z_means):
    """
    SHist论文中Algorithm4的Sever端部分，将z_means舍入得到y，对y解码得到
    :param z_means:z向量平均
    :return
    """
    m=len(z_means)
    z_means[z_means>=0]=1/(np.sqrt(m))
    z_means[z_means<0]=-1/(np.sqrt(m))
    y=z_means
    v_estimate=decoder(y)
    code_v=encoder(v_estimate)
    f_v_estimat=np.dot(code_v,z_means)
    return v_estimate,f_v_estimat




