# Python杂记

## 一些库&方法

### 1.copy&deepcopy

字面意思：浅拷贝&深拷贝

用法：

```python
a = {1, 2, 3}

b = a.copy()
```

a和b是两个独立的对象，拷贝父对象，但是都指向同一个子对象。

```python
import copy

a = {1, 2, 3}

b = copy.deepcopy(a)
```

a和b是两个独立的对象，完全拷贝了父对象及其子对象，并且两者完全独立，即产生了两个完全分离的子对象。