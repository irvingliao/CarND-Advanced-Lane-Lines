import numpy as np
from collections import deque
from Lane import Line, Lane

def addNewObj(queue, x):
    queue.append(x)

q = deque(maxlen=5)

addNewObj(q, np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02]))
addNewObj(q, np.array([ 4.13935315e-04, -6.77507980e-01,  8.76902175e+02]))
addNewObj(q, np.array([ 8.13935315e-04, -1.77507980e-01,  9.76902175e+02]))

total = np.array([0,0,0], dtype='float')
for x in q:
    total += x

avg = total/len(q)
print('average:', avg)

# q.append(2)
# q.append(3)
# q.append(4)
# q.append(5)
# q.append(10)

# print(q)
# print(q[-1])

# lanes = []
# lane1 = Lane()
# lane1.left_line.bestx = 200
# lane1.left_line.current_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])

# lane1.right_line.bestx = 900
# lane1.right_line.current_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

# lane2 = Lane()
# lane2.left_line.bestx = 300
# lane2.left_line.current_fit = np.array([ 3.13935315e-04, -4.77507980e-01,  5.76902175e+02])

# lane2.right_line.bestx = 800
# lane2.right_line.current_fit = np.array([2.17622148e-04, -5.93848953e-01,  7.11806170e+03])

# lanes.append(lane1)
# lanes.append(lane2)

# lanes[0].toString()
# print("")
# lanes[1].toString()
