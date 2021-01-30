# pytorch的gather和scatter_两个用法。

* Example:
表一

|一班|二班|三班|四班|
|---|----|---|---|
|小红|校花|小刚|小赵|
|小明|小芳|小强|小王|

此处假设有四个班级，每个班级有两个学生，现在他们要报高考志愿：清华、北大、负担、上交、浙大，序号依次
为0-4.各班的报名状况为：

表2

|一班|二班|三班|四班|
|---|----|---|---|
|0(小红)|1(校花)|4(小刚)|3(小赵)|
|2(小明)|3(小芳)|0(小强)|1(小王)|
如果我想知道有哪些人报了清华大学，他们分别在哪个班，上面的表格还是不够直观。如果生成一个把大学作为行、把
班级作为列的表格，则可以清晰地看到各个大学在各班的报名情况：

表3


|   |一班|二班|三班|四班|
|---|----|---|---|---|
|清华0|0(小红)| |0(小强)||
|北大1   | |1(小华)| |1(小王)|
|复旦2   |2(小明)| | | |
|上交3   | |3(小芳)| |3(小赵)|
|浙大4   | | |4(小刚)| |

**到了这里，已经完成了一次pytorch中的scatter操作。**

## scatter
````python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
````
先从简单的二维情况分析，把上面的例子第一行改成二维的
````python
self[index[i][j]][j] = src[i][j] # if dim == 0

````
然后，类比于报志愿的问题，src是表1，index是表2，self是表3.i是学生在该班级内部的编号(所在行),j是
班级编号,注意i和j都从0开始.

那么Src[i][j]表示的是j班的第i个同学.根据表1 src[0][0]就是小红

index[i][j]表示的是j班i号同学所报大学的编号。因此 ，显而易见的是index的形状应该和src一致，（严格来讲，只需要
index各维度尺寸小于src的）。同时index表中最大的数，不应该超出可选大学的最大下表，也就是self的末行下标。

gather所做的操作，先查出index[i][j] 找到第j班i同学所报的大学编号，该编号就是需要修改的表3的
行下标，由于该同学所在的班级是j班，因此需要修改表三的列下标是j；确定了需要修改的表三的行、列下标，
接下来需要确定的是修改的值。在我们的类比中，src[i][j]是该同学的名字，因此把名字填入self中对应的
位置即可。

代码
````python
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
````
如果把代码中的dim=0改成dim=1, scatter_的赋值规则变成了：
````python
self[i][index[i][j]] = src[i][j] # if dim == 1
````
这个时候index存储的是self的列号，即大学的序号变成列了。i是班级序号，j是学生在该班级的序号。
最后得到的self为表三的转置。

至于3D tensor，则可以扩展类比，增加一个高中学校的编号。
````python
self[index[i][j][k]][j][k] = src[i][j][k] # if dim == 0

index[i][j][k]: k:高中，j班，i号学生所报高校编号

self[index[i][j][k]][k][j]：index[i][j][k]高校所招的k号高中j班学生的名字

````
````python
scatter_(dim, index, src, reduce=None) → Tensor
index存储的是self中 dim维的下标。scatter_的做法是,把src中index所覆盖的元素,填充到self中。其中
dim维下标由该元素对应的index确定。
```` 

##gather
````python
torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor

out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
````
其中 input为表三，index为表二，out为表一。即：scatter是由表一表二得到表三，而gather则是由表三biaoer得到表一
gather是scatter的逆运算

gather所做的事情就是，知道了各班各同学报的学校的下标（index）,也知道各学校在各班的招生情况(input)
反推哪个班有哪一个同学。

````python
input = torch.gather(tgt, 0, index)
print(input)
# tensor([[0., 2., 4., 6.],
#         [1., 3., 5., 7.]])
````
**总结**
· scatter的目标是修改现有矩阵，gather是根据index从现有矩阵提取元素，构成一个新矩阵。新矩阵的形状
与index形状一致。



