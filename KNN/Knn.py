from math import sqrt
from collections import Counter
import numpy as np

dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new=[5,7]
def k_nearest(data, predict,k=3):
    distance=[]
    for group in data:
        for features in data[group]:
            dis=np.linalg.norm(np.array(features)-np.array(predict))
            distance.append([dis,group])
    votes=[ i[1] for i in sorted(distance)[:k]]
    print(Counter(votes).most_common(1)[0][0])
    return Counter(votes).most_common(1)[0][0]

result=k_nearest(dataset,new)
