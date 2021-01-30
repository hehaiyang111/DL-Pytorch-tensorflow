import torch

# 生成表1，存的是各班学生的姓名，用序号代替，序号按照班级依次递增
src = torch.arange(8).view(4,2).transpose(0,1).type(torch.float32)
print(src)
# tensor([[0., 2., 4., 6.],
#         [1., 3., 5., 7.]])

# 生成表2，存的是各班各同学所报大学的下标
index = torch.LongTensor([[0, 1, 4, 3], [2, 3, 0, 1]])
print(index)
# tensor([[0, 1, 4, 3],
#         [2, 3, 0, 1]])

# 初始化表3 即各校报名情况。
tgt = torch.ones(5, 4) * -1
print(tgt)
# tensor([[-1., -1., -1., -1.],
#         [-1., -1., -1., -1.],
#         [-1., -1., -1., -1.],
#         [-1., -1., -1., -1.],
#         [-1., -1., -1., -1.]])

# dim=0代表index存储的数字为tgt的行号。
tgt.scatter_(dim=0, index=index, src=src)
print(tgt)
# tensor([[ 0., -1.,  5., -1.],
#         [-1.,  2., -1.,  7.],
#         [ 1., -1., -1., -1.],
#         [-1.,  3., -1.,  6.],
#         [-1., -1.,  4., -1.]])

input = torch.gather(tgt, 0, index)
print(input)
# tensor([[0., 2., 4., 6.],
#         [1., 3., 5., 7.]])
