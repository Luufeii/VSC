# 这里把需要的文本txt信息写好
from M_attri import Att

attri = Att()
attri.compute_mul_defect_att()

# print(attri.single_defect_att.keys())
# print(len(attri.single_defect_att.keys()))
# print(attri.mul_defect_att.keys())
# print(len(attri.mul_defect_att.keys()))
# print(attri.total_defect_att.keys())
# print(len(attri.total_defect_att.keys()))


with open("all_classes.txt", "w") as f:
    for i, cla in enumerate(attri.total_defect_att.keys()):
        ret = ""
        ret += str(i+1)
        ret += " "
        ret += str(cla)
        f.write(ret)
        f.write('\n')

with open("all_attribute.txt", "w") as f:
    for att in attri.total_defect_att.values():
        ret = ""
        for y in att:
            ret += str(int(y))
            ret += " "
        f.write(ret)
        f.write('\n')

with open("train_classes.txt", "w") as f:
    for cla in attri.single_defect_att.keys():
        ret = ""
        ret += str(cla)
        f.write(ret)
        f.write('\n')

with open("test_classes.txt", "w") as f:
    for cla in attri.mul_defect_att.keys():
        ret = ""
        ret += str(cla)
        f.write(ret)
        f.write('\n')
