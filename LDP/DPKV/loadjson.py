"""
@author:FZX
@file:loadjson.py
@time:2020/5/15 9:54 上午
"""
import jsonlines



json_filename = '../LDPdataset/Clothing/data/Clothing.json'  # 这是json文件存放的位置
txt_filename = '../LDPdataset/Clothing/data/Clothing.txt'  # 这是保存txt文件的位置
file = open(txt_filename, 'w')
with open(json_filename) as f:
    for pop_dict in jsonlines.Reader(f):
        userid=pop_dict['user_id']
        itemid = pop_dict['item_id']
        rating = pop_dict['rating']
        temp = str(userid)+','+str(itemid) + ',' + str(rating)
        file.write(temp + '\n')
    file.close()
