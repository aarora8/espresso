# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import numpy as np
list_a = []
for i in range(2):
    for j in range(5):
        list_a.append(i)

list_a = np.random.permutation(list_a)
print('class labels')
print(list_a)
list_a = np.array(list_a)


index_i = 0
classid_of_index0=list_a[index_i]
print('class_of_index0: ', classid_of_index0)
classid_of_index0_locations = np.where(list_a == classid_of_index0)
classid_of_index0_locations = classid_of_index0_locations[0]
print('class_of_index0_locations', classid_of_index0_locations)
print(classid_of_index0_locations != index_i)
same_index_list = classid_of_index0_locations[classid_of_index0_locations != index_i]
print(same_index_list)
print(same_index_list[0:2])

num_tokens_vec = [5,6,7,5,4,3,5,4,6,7]
for pos in same_index_list[0:2]:
    print(num_tokens_vec[pos])
max_val = tuple(num_tokens_vec[pos] for pos in same_index_list[0:2])
max_val1 = max(max_val)
print(max_val)
print(max_val1)
