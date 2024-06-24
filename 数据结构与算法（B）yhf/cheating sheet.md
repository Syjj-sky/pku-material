

# Fraction

```python
class Fraction:
	def __init__(self,top,bottom):
		self.num=top
    	self.den=bottom
    def __str__(self):#print对象时自动调用
        return str(self.num)+'/'+str(self.den)
	def __add__(self,otherfraction):
        new_num=self.num*otherfraction+self.den*otherfraction.num
        new_den=self.den*otherfraction.den
		return Fraction(new_num//gcd,new_den//gcd)   
    def __eq__(self,other):
        num_1=self.num*other.den
        num_2=other.num*self.den
        return num_1 == num_2
```



# 链表

## 约瑟夫问题

```python
###循环列表：考试-直接除余法
n,p,m=map(int,input().split())#n个元素，p为起始元素，报m的元素删除
ans=[]
kid=[1]*n#初始循环列表，报数后改为0
x,i=p-2+n,0#x记录循环孩子，i记录报数孩子
for _ in range(n):#删除n个孩子
    while True:
        x=(x+1)%n#下一个元素
        if kid[x]:
            i+=1
            if i%m == 0:
                ans.append(str(x+1))
                kid[x]=0
                break
print(','.join(ans))
```

## Deque

```python
from collections import deque
d=deque(['a','b','c','d'])
d.appendleft('e');d.popleft('e')
rd=reversed(d);d.rotate
```

## 链表检测环

快慢指针

```python
def has_loop(head):#head->class Node
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

# 排序

## 归并排序

### 逆序数

逆序数：Ultra-Sort排序，交换相邻元素到顺序的最小交换次数/在将L和R的元素分别排入arr时，若排入L元素，逆序对不增加，排入R元素，L中剩余元素个增加一对逆序对。

```python
pair=0
def mergeSort(arr):
    global pair#逆序对数
    if len(arr) > 1:
        mid=len(arr)//2
        L=arr[:mid]
        R=arr[mid:]
        mergeSort(L)
        mergeSort(R)
        i,j,k=0,0,0
        #将排好序的L和R按大小排入arr中
        while i < mid and j < (len(arr)-mid):
            if L[i] <= R[j]:
                arr[k]=L[i]
                i+=1
            else:
                pair+=(mid-i)
                arr[k]=R[j]
                j+=1
            k+=1
        #填入没有遍历到的L或R
        while i < len(L):
            arr[k]=L[i]
            i+=1;k+=1
        while j < len(R):
            arr[k]=R[j]
            j+=1;k+=1
    return pair
```

## Heapq

```python
import heapq
a=[1,3,4,2]
heapq.heapqify(a)#heapq库函数可直接对list操作
heapq.heappush(a,0)
x=heapq.heappop(a)#0
heapq.heapreplace(a,5)#[5,3,4,2]
heapq.heappushpop(a,-1)#[5,3,4,2]
```

# 栈

## 快速堆猪（日志分析）

操作：插入值，删除最后插入值，查询最大值

### 懒删除/辅助栈维护最大值

```python
pig=[]
while True:
    try:
        x=input()
    except EOFError:
        break
    if x == 'min':
        if pig:
            print(pig[-1])
    else:
        if x == 'pop':
            if pig:pig.pop()
        else:
            n=int(x.split()[1])
            if pig:
                pig.append(min(int(n),pig[-1]))
            else:
                pig.append(int(n))
```

## 单调栈

单调栈：保持某种最优状态（比如栈顶最大，车处于最快状态），问题在于考虑stack.pop()条件。另外输出时考虑不断输出栈顶还是只用输出最后单调栈内元素

### 模板

存贮单调递增/递减的栈

1 3 2 5 8 1 5 4 输出右边第一个大于x的元素下标

2 4 4 5 0 7 0 0

```python
n=int(input())
num=list(map(int,input().split()))
ans=[0]*n
stack=[]
for i in range(n-1,-1,-1):#从右往左算
    while stack and num[stack[-1]] <= num[i]:#<=x的元素删去
        stack.pop()
    if stack:
        ans[i]=stack[-1]+1#存贮右边第一个大于x的元素下标
    stack.append(i)#得到单调递减的stack
print(*ans)
```

### bisect&单调栈

https://www.luogu.com.cn/record/161212441

注意：问题不同，一个求l-n序列中最大值，只需存单调栈即可，一个求x后最近的最大值，需要求暂时的stack[-1]

```python
import bisect
m,d=map(int,input().split())
t,l=0,0
f,id=[],[]
for _ in range(m):
    op,n=input().split()
    n=int(n)
    if  op == 'A':
        n=(n+t)%d
        while f and f[-1]< n:
            f.pop();id.pop()
        f.append(n);id.append(l)#id存贮单调递减单调栈
        l+=1
    else:
        t=f[bisect.bisect_left(id,l-n)]#bisect用于查找两个列表f和id对应位置的元素
        print(t)
```

### 赛车：贪心&单调栈

https://www.luogu.com.cn/record/161238582

```python
class Car:
    def __init__(self,v,k,id):
        self.v,self.k,self.id=v,k,id
def time(a,b):#b追上a的时间
    if a.v == b.v:
        return float('inf')
    else:
        return (a.k-b.k)/(b.v-a.v)
n=int(input())
cars=[Car(0,0,i)for i in range(1,n+1)]
for i,k,v in zip(range(n),input().split(),input().split()):
    cars[i].v,cars[i].k=int(v),int(k)
cars.sort(key=lambda x:(x.v,x.k))
stack=[]
for i in range(n):
    car=cars[i]
    while stack:
        if car.k > stack[-1].k:
            stack.pop()
        elif len(stack) > 1 and time(stack[-1],car) < time(stack[-2],stack[-1]):
            stack.pop()
        else:break
    stack.append(car)
print(len(stack))
print(*sorted([car.id for car in stack]))
```



## 表达式转换

### 波兰表达式/前缀

求值

```python
num=[]
for i in input().split()[::-1]:
    if i in '+-*/':
        a,b=num.pop(),num.pop()
        num.append(str(eval(a+i+b)))
    else:
        num.append(i)
print('%.6f'%float(num[0]))
```

# 树

## 遍历树

```python
def preorder(tree):#前序遍历
    if tree:
        print(tree.get_root)
        preorder(tree.get_l)
        preorder(tree.get_r)
```



## 二叉树

```python
def build(n):
    tree=[Node(i)for i in range(n)]#用列表建树
    check=set()
    for i in range(n):
        l,r=map(int,input())
        if l != -1:
            tree[i].left=tree[l]
            check.add(l)
        if r != -1:
            tree[i].right=tree[r]
            check.add(r)
    return tree[*set(range(n))-check]#返回根节点
def height(root):#计算树的高度
    if root is None:
        return 0
    return max(height(root.left),heigt(root.right))+1
def leaf(root):#结算叶子结点数
    if root is None:
        return 0
    if root.left == None and root.right == None:
        return 1
    return leaf(root.left)+leaf(root.right)
```

## 二叉搜索树

```python
def build_tree(node,x):
    if node == None:
        return TreeNode(x)
    else:
        if x < node.key:
            node.left=build_tree(node.left,x)
        else:
            node.right=build_tree(node.right,x)
    return node
for x in arr:
    build_tree(root,x)
```



## 表达式-表达式树-表达式求值

```python
class Node:
    def __init__(self,key):
        self.key=key
        self.left,self.right=None,None
def post(arr):#中缀表达式转后缀表达式
    stack=[]
    result=[]
    op={'+':1,'-':1,'*':2,'/':2}
    for x in arr:
        if x.isalpha():result.append(x)
        elif x == '(':stack.append(x)
        elif x == ')':
            while stack and stack[-1] != '(':
                result.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] in op and op[x] <= op[stack[-1]]:
                result.append(stack.pop())
            stack.append(x)
    while stack:
        result.append(stack.pop())
    return result
def build(arr):#后缀表达式建树
    stack=[]
    for x in arr:
        node=Node(x)
        if x in '+-*/':
            node.right=stack.pop()
            node.left=stack.pop()
        stack.append(node)
    return stack[0]
def depth(root):
    if not root:
        return 0
    return max(depth(root.left),depth(root.right))+1
def Print(root,d):#输出二叉树
    if d == 0:
        return root.key
    graph = [' '*(2**d-1)+root.val+' '*(2**d-1)]
    graph.append(' '*(2**d-2)+('/' if root.left is not None else ' ')
                 +' '+('\\' if root.right is not None else ' ')+' '*(2**d-2))
    d -= 1
    l = Print(root.left, d) if root.left is not None else [' '*(2**(d+1)-1)]*(2*d+1)
    r = Print(root.right, d) if root.right is not None else [' '*(2**(d+1)-1)]*(2*d+1)
    for i in range(2*d+1):
        graph.append(l[i]+' '+r[i])
    return graph
def cal(root):#计算后缀表达式树的值
    if root.val.isalpha():
        return my_dict[root.val]
    else:
        lv,rv=cal(root.left),cal(root.right)
        return int(eval(str(lv)+root.val+str(rv)))
arr=list(input())#输入中缀表达式
arr=post(arr)#转后缀表达式
root=build(arr)#后缀表达式建树
d=depth(root)#树的深度
Print(root,d)#按格式输出
my_dict={}#输入各变量的值
print(cal(root))#输出后缀表达式树的值
```



## 二叉堆

```python
class BinHeap:
    def __init__(self):#初始化堆列表和size
        self.heaplist=[0]
        self.size=0
    def insert(self,x):#插入x到最后并从下往上调整
        self.heaplist.append(x)
        self.size+=1
        self.percUp(self.size)
    def delMin(self):#删除最小(heaplist[1])元素并从上往下调整
        re=self.heaplist[1]
        self.heaplist[1]=self.heaplist[self.size]
        self.size-=1
        self.heaplist.pop()
        self.percDown(1)
        return re
    def percUp(self,i):#从下往上调整
        while i//2 > 0:
            if self.heaplist[i] < self.heaplist[i//2]:
                self.heaplist[i],self.heaplist[i//2]=self.heaplist[i//2],\
                    self.heaplist[i]
            i=i//2
    def percDown(self,i):#从上往下调整
        while (i*2) <= self.size:
            mc=self.minChild(i)
            if self.heaplist[i] > self.heaplist[mc]:
                self.heaplist[i],self.heaplist[mc]=self.heaplist[mc],\
                    self.heaplist[i]
                i=mc
            else:
                break
    def minChild(self,i):#判断左右最小子节点
        if i*2+1 >self.size:
            return i*2
        elif self.heaplist[i*2] < self.heaplist[i*2+1]:
            return i*2
        else:
            return i*2+1
    def buildHeap(self,alist):#从元素列表构建堆
        i=len(alist)//2
        self.size=len(alist)
        self.heaplist=[0]+alist[:]
        while i > 0:
            self.percDown(i)
            i-=1
```

## AVL树

```python
class Node():
    def __init__(self,key):
        self.key=key
        self.left,self.right=None,None
        self.h=1
class AVL():
    def __init__(self):
        self.root=None
    def insert(self,value):
        if self.root:
            self.root=self._insert(value,self.root)
        else:self.root=Node(value)
    def _insert(self,value,node):
        if not node:
            return Node(value)
        elif value < node.key:
            node.left=self._insert(value, node.left)
        else:
            node.right=self._insert(value, node.right)
        node.h=1+max(self._height(node.left),self._height(node.right))
        b=self.balance(node)
        if b > 1:
            if value < node.left.key:#树形为LL
                return self._rotate_r(node)
            else:#树形为LR
                node.left=self._rotate_l(node.left)
                return self._rotate_r(node)
        if b < -1:
            if value > node.right.key:#树形为RR
                return self._rotate_l(node)
            else:	# 树形是 RL
                node.right = self._rotate_r(node.right)
                return self._rotate_l(node)
        return node
    def _height(self,node):
        if not node:
            return 0
        else:
            return node.h
    def balance(self,node):#计算节点node的平衡因子
        if not node:
            return 0
        return self._height(node.left)-self._height(node.right)
    def _rotate_r(self,node):
        x,y=node,node.left
        x.left,y.right=y.right,x
        x.h=1+max(self._height(x.left),self._height(x.right))
        y.h=1+max(self._height(y.left),self._height(y.right))
        return y
    def _rotate_l(self,node):
        x,y=node,node.right
        x.right,y.left=y.left,x
        x.h=1+max(self._height(x.left),self._height(x.right))
        y.h=1+max(self._height(y.left),self._height(y.right))
        return y
n=int(input())
avl=AVL()
for value in map(int,input().split()):
    avl.insert(value)
```

![](D:\北大\课程\24春\数算\cheating sheet\avl树旋转.jpg)

## Huffmann编码树

[OpenJudge - 22161:哈夫曼编码树](http://cs101.openjudge.cn/practice/22161/)

```python
import heapq
class Node():
    def __init__(self,char,freq):
        self.key=char
        self.freq=freq
        self.left,self.right=None,None
    def __lt__(self,other):
        return self.freq < other.freq
    
def build_huff():#构建哈夫曼树
    n=int(input())
    chars=[]#存贮字母节点的最小堆
    for _ in range(n):
        char,freq=input().split()#各字母及使用频率
        chars.append(Node(char,int(freq)))
    heapq.heapify(chars)
    while len(chars) > 1:
        a,b=heapq.heappop(chars),heapq.heappop(chars)#合并频率最低的两个字母
        parent=Node(a.key+b.key,a.freq+b.freq)
        parent.left,parent.right=a,b
        heapq.heappush(chars,parent)
    return chars[0]
```

### 并查集

```python
class Union:
    def __init__(self,n):
        self.parent=list(range(n))
        self.size=[1]*n
    def find(self,i):
    	if self.parent[i] != i:
        	self.parent[i]=self.find(self.parent[i])
    	return self.parent[i]
    def union(self,x,y):
        xp,yp=self.find(x),self.find(y)
        if xp == yp:
            return
        xsize,ysize=self.size[x],self.size[y]#合并规则：按照大小合并
        if xsize < ysize:
            self.parent[xp]=yp
            self.size[yp]+=self.size[xp]
        else:
            self.parent[yp]=xp
            self.size[xp]+=self.size[yp]
```

# 图

## bfs

### 模板

标记走过的节点/方向：1.visited记录 2.memo记录，判断所需时间/代价是否最小（设置初始值inf，Robot，鸣人与佐助...）

```python
from collections import deque
parent={}
graph={}
def bfs(start,end):#返回值为step步数
    quene=deque([start])
    visited=set()
    #memo={node:float('inf') for node in graph},例如：记录每点的step
    while quene:
        node,step=quene.popleft()
        if node[0] == end[0]:
            return step
            break
        if node not in visited:
            visited.append(node)
            for next in graph[node]:
                parent[next]=node
                quene.append((next,step+1))
    return 'impossible'
ans=[]
def back(end,start):
    ans.append(end)
    if end == start:
        return
    else:
        back(parent[end],start)
```

#### Saving the Monk

```python
from queue import PriorityQueue
class Node:
    def __init__(self, x, y, t, k, s):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        self.s = s
    def __lt__(self, other):
        return self.t < other.t
def bfs(maze, n, m):
    x0, y0, count = 0, 0, 0
    lst = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if maze[i][j] == 'K':
                x0, y0 = i, j
            if maze[i][j] == 'S':
                lst[i][j] = count
                count += 1
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    memo = [[[float('inf') for _ in range(m+1)] for _ in range(n)] for _ in range(n)]
    queue = PriorityQueue()
    queue.put(Node(x0, y0, 0, 0, 0))
    memo[x0][y0][0] = 0
    while not queue.empty():
        node = queue.get()
        if maze[node.x][node.y] == 'T' and node.k == m:
            return node.t
        for a, b in dir:
            nx, ny = node.x+a, node.y+b
            if 0 <= nx < n and 0 <= ny < n:
                if maze[nx][ny] == '#':
                    continue
                elif maze[nx][ny] == 'S':
                    if (node.s >> lst[nx][ny]) & 1:
                        if node.t+1 < memo[nx][ny][node.k]:
                            memo[nx][ny][node.k] = node.t+1
                            queue.put(Node(nx, ny, node.t+1, node.k, node.s))
                    else:
                        if node.t+2 < memo[nx][ny][node.k]:
                            memo[nx][ny][node.k] = node.t+2
                            queue.put(Node(nx, ny, node.t+2, node.k, node.s | (1 << lst[nx][ny])))
                elif maze[nx][ny].isdigit():
                    if int(maze[nx][ny]) == node.k+1:
                        if node.t+1 < memo[nx][ny][node.k+1]:
                            memo[nx][ny][node.k+1] = node.t+1
                            queue.put(Node(nx, ny, node.t+1, node.k+1, node.s))
                    else:
                        if node.t+1 < memo[nx][ny][node.k]:
                            memo[nx][ny][node.k] = node.t+1
                            queue.put(Node(nx, ny, node.t+1, node.k, node.s))
                else:
                    if node.t+1 < memo[nx][ny][node.k]:
                        memo[nx][ny][node.k] = node.t+1
                        queue.put(Node(nx, ny, node.t+1, node.k, node.s))
    return 'impossible'
while True:
    n, m = map(int, input().split())
    if n == m == 0:
        break
    maze = [list(input()) for _ in range(n)]
    print(bfs(maze, n, m))
```

## dfs

### 模版

```python
visited={start}
path=[start]
def dfs(start):
    global path
    if start == end:
        print(path)
        return
    for next in graph[start]:
        if next not in visited:
            visited.add(next)
            path.append(path)
            dfs(next)
            path.pop()
```

### 无向图判断连通/环

```python
from collections import defaultdict
def isconnected(graph,n):
    visited=set()
    def dfs(init):
        visited.add(init)
        for next in graph[init]:
            if next not in visited:
                dfs(next)
    dfs(0)
    return [False,True][len(visited) == n]
    
def isloop(graph,n):
    visited=set()
    def dfs(init,last):
        visited.add(init)
        for next in graph[init]:
            if next in visited:
                if next != last:
                    return True
                else:continue
            else:
                if dfs(next,init):
                    return True
        return False
    for node in graph.keys():
        if node not in visited:
            if dfs(node,None):
                return True
    return False
```

### 有向图判断回路（拓扑排序）

舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/

```python
from collections import defaultdict
from queue import Queue   #FIFO
def build_graph(n,m):
    graph=defaultdict(list)
    for _ in range(m):
        x,y=map(int,input().split())
        graph[x].append(y)
    return graph

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = Queue()
    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.put(u)
    # 执行拓扑排序
    while not queue.empty():
        u = queue.get()
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.put(v)
    # 检查是否存在环
    if len(result) == len(graph):
        return 'No'
    else:
        return 'Yes'
t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    graph=build_graph(n,m)
    print(topological_sort(graph))
```

### 剪枝

- **最优性剪枝**（发现已经次优于当前最优，那么就可以放弃后续搜索）

- **可行性剪枝**（判断是否违反了问题的约束条件）

- **启发式剪枝**（提前预测哪些分支很可能不会产生最优解，从而加速搜索）

- **对称性剪枝**（利用对称性剪枝）

- **记忆化剪枝**（记录先前搜索过的状态进行判重）

  记忆化搜索

  1.dfs+visited字典

  2.lru_cache

  ```python
  #斐波那契数列
  from functools import lru_cache
  @lru_cache(maxsize=None)
  def f(x):
      if x == 0 or x == 1:
          return 1
      return f(x-1)+f(x-2)
  n=int(input())
  print(f(n))
  ```

  

#### Warnstorff算法

dfs在每次选择是选择可行节点最少的子节点，最快得到答案

骑士周游

```python
def route(point):#存贮走法
    x,y=point
    return [(x+i,y+j)for i,j in dir if 0 <= x+i < n and 0 <= y+j < n and mat[x+i][y+j]]

ans,cnt=0,1
def dfs(x,y):
    global cnt
    ans=False
    if cnt == n*n:#马走日问题，走完所有格子
        return True
    for xx,yy in sorted(route((x,y)),key=lambda s:len(route(s))):#先走周围格子能走格子少的格子，剪枝
            cnt+=1
            mat[xx][yy]=0
            ans=dfs(xx,yy)
            if ans:
                break
            cnt-=1#回溯
            mat[xx][yy]=1
    return ans
n=int(input())
x,y=map(int,input().split())
mat={i:{j:1 for j in range(n)} for i in range(n)}
dir=[(i,j) for i,j in zip([1,1,-1,-1,2,2,-2,-2],[2,-2,2,-2,1,-1,1,-1])]
mat[x][y]=0
cnt=1
if dfs(x,y):
    print('success')
else:
    print('fail')
```

## Dijk算法

1.初始化数组

2.当前节点--未访问节点中距离最小

3.更新当前节点距离

4.标记当前访问

```python
import heapq
def bfs(x,y):#从x，y开始搜索
    route=[(0,x,y)]
    visited={(x,y)}
    while route:
        get,x,y=heapq.heappop(route)#权值最小的路径
        visited.add((x,y))
        if x == x1 and y == y1:#到达目的地x1，y1
            return get
        for dx,dy in dir:
            xx,yy=x+dx,y+dy
            if 0 <= xx < m and 0 <= yy < n and (xx,yy) not in visited and ma[xx][yy] != '#':#满足条件，添加（xx，yy）
                g=get+weight[((x,y),(xx,yy))]#从xx到yy的权值
                heapq.heappush(route,(g,xx,yy))
    return 'NO'

m,n,p=map(int,input().split())
ma={i:{j:x for j,x in zip(range(n),input().split())}for i in range(m)}
dir=[[0,-1],[0,1],[1,0],[-1,0]]
for _ in range(p):
    x,y,x1,y1=map(int,input().split())
    if ma[x][y] == '#' or ma[x1][y1] == '#':
        print('NO')
    else:
        print(bfs(x,y))
```

# 最小生成树

## Prim算法

```python
def cmp(key1, key2):
    return (key1, key2) if key1 < key2 else (key2, key1)
def prim(graph, init_node):
    visited = {init_node}
    candidate = set(graph.keys())
    candidate.remove(init_node)  # add all nodes into candidate set, except the start node
    tree = []
    while len(candidate) > 0:
        edge_dict = dict()#存贮边
        for node in visited:
            for connected_node, weight in graph[node].items():
                if connected_node in candidate:
                    edge_dict[cmp(connected_node, node)] = weight
        edge, cost = sorted(edge_dict.items(), key=lambda kv: kv[1])[0]  # 权重最小边
        tree.append(edge)
        visited.add(edge[0]);visited.add(edge[1])
        candidate.discard(edge[0]);candidate.discard(edge[1])
    return tree
graph_dict = {
    "A": {"B": 7, "D": 5},
    "B": {"A": 7, "C": 8, "D": 9, "E": 5},
    "C": {"B": 8, "E": 5},
    "D": {"A": 5, "B": 9, "E": 15, "F": 6},
    "E": {"B": 7, "C": 5, "D": 15, "F": 8, "G": 9},
    "F": {"D": 6, "E": 8, "G": 11},
    "G": {"E": 9, "F": 11}
}
path = prim(graph_dict, "D")
print(path)  # [('A', 'D'), ('D', 'F'), ('A', 'B'), ('B', 'E'), ('C', 'E'), ('E', 'G')]
```

## Kruskal算法

```python
def cmp(key1, key2):
    return (key1, key2) if key1 < key2 else (key2, key1)
def find_parent(record, node):
    if record[node] != node:
        record[node] = find_parent(record, record[node])
    return record[node]
def naive_union(record, edge):
    u, v = find_parent(record, edge[0]), find_parent(record, edge[1])
    record[u] = v
def kruskal(graph, init_node):
    edge_dict = {}
    for node in graph.keys():
        edge_dict.update({cmp(node, k): v for k, v in graph[node].items()})
    sorted_edge = list(sorted(edge_dict.items(), key=lambda kv: kv[1]))
    tree = []
    connected_records = {key: key for key in graph.keys()}
    for edge_pair, _ in sorted_edge:
        if find_parent(connected_records, edge_pair[0]) != \
                find_parent(connected_records, edge_pair[1]):
            tree.append(edge_pair)
            naive_union(connected_records, edge_pair)
    return tree
graph_dict = {
    "A": {"B": 7, "D": 5},
    "B": {"A": 7, "C": 8, "D": 9, "E": 5},
    "C": {"B": 8, "E": 5},
    "D": {"A": 5, "B": 9, "E": 15, "F": 6},
    "E": {"B": 7, "C": 5, "D": 15, "F": 8, "G": 9},
    "F": {"D": 6, "E": 8, "G": 11},
    "G": {"E": 9, "F": 11}
}
path = kruskal(graph_dict, "D")
print(path)  # [('A', 'D'), ('D', 'F'), ('A', 'B'), ('B', 'E'), ('C', 'E'), ('E', 'G')]
```

# 拓扑排序

有向无环图拓扑排序

https://www.luogu.com.cn/problem/P1038

## Kahn算法

```python
from collections import defaultdict
from queue import Queue   #FIFO
def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = Queue()
    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.put(u)
    # 执行拓扑排序
    while not queue.empty():
        u = queue.get()
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.put(v)
    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None
```



# 背包问题

https://blog.csdn.net/m0_62861576/article/details/137016699

## 01背包

背包容积：v

n个物品，物品i体积wi，价值wi，每件物品拿1件/不拿

初始化：恰好装满，仅dp [n] [0] 为0，其他为-inf

​		最大：均为0

```python
n,v=map(int,input().split())
dp=[[0]*(v+1)for _ in range(n+1)]
for i in range(1,n+1):
    wi,vi=map(int,input().split())
    for j in range(0,v+1):
        if j < wi:
            dp[i][j]=dp[i-1][j]
        else:
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-wi]+vi)
print(dp[n][v])
#滚动数组优化
n,v=map(int,input().split())
dp=[[0]*(v+1)]
for i in range(1,n+1):
    wi,vi=map(int,input().split())
    for j in range(v,wi-1,-1):
        dp[j]=max(dp[j],dp[j-wi]+vi)
print(dp[v])
```

## 完全背包

背包容积：v

n个物品，物品i体积wi，价值wi，每件物品拿任意件/不拿

```python
n,v=map(int,input().split())
dp=[[0]*(v+1)for _ in range(n+1)]
for i in range(1,n+1):
    wi,vi=map(int,input().split())
    for j in range(0,v+1):
        if j < wi:
            dp[i][j]=dp[i-1][j]
        else:
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-wi]+vi)
print(dp[n][v])
#滚动数组优化
n,v=map(int,input().split())
dp=[[0]*(v+1)]
for i in range(1,n+1):
    wi,vi=map(int,input().split())
    for j in range(wi,v+1):
        dp[j]=max(dp[j],dp[j-wi]+vi)
print(dp[v])
```

## 多重背包

背包容积：v

n个物品，物品i体积wi，价值wi，每件物品拿有限s件/不拿

```python
n,v=map(int,input().split())
dp=[[0]*(v+1)for _ in range(n+1)]
for i in range(1,n+1):
    w,v,s=map(int,input().split())
    for j in range(v+1):
        for k in range(min(s,j//w)+1):
            dp[i][j]=max(dp[i][j],dp[i-1][j-k*w]+k*v)
print(dp[n][v])
```

## 二维背包

背包容积：v 背包称重：m

n个物品，物品i体积wi，价值wi，重量mi，每件物品拿1件/不拿

```python
#滚动数组优化
n,v,m=map(int,input().split())
dp=[[0]*(m+1)for _ in range(v+1)]
for i in range(1,n+1):
    wi,mi,vi=map(int,input().split())
    for j in range(v,wi-1,-1):
        for k in range(m,mi-1,-1):
            dp[j][k]=max(dp[j][k],dp[j-vi][k-mi]+wi)
print(dp[v][m])
```

## 分组背包

背包容积：v 

n组物品，物品组i：si件，每件体积wik，价值vik，每组最多拿1件

```python
n,v=map(int,input().split())
dp=[[0]*(v+1)for _ in range(n+1)]
for i in range(1,n+1):
    group=[]
    s=int(input())
    for _ in range(s):
        w,v=map(int,input().split())
        for j in range(v+1):
            if j< w:
                dp[i][j]=max(dp[i][j],dp[i-1][j])
            else:
                dp[i][j]=max(dp[i][j],dp[i-1][j],dp[i-1][j-w]+v)
print(dp[n][v])
```

# 计概

## asc2

48-0 65-A 97-a

## bisect

![image-20240605134356003](C:\Users\sky\AppData\Roaming\Typora\typora-user-images\image-20240605134356003.png)

## 进制转换：bin(int(n,16))(16→10→2)

## prime

```python
def era(n,prime):#埃氏筛
    p=2
    while (p*p <= n):
        if (prime[p] == True):
            for i in range(p*2,n+1,p):
                prime[i]=False
        p+=1
def euler(r):#欧拉筛
    prime=[0 for i in range(r+1)]
    common=[]
    for i in range(2,r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j]=1
            if i%j == 0:
                break
    return prime
```

## re

![image-20240605135226086](C:\Users\sky\AppData\Roaming\Typora\typora-user-images\image-20240605135226086.png)
