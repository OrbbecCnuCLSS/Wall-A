import os

cnt = 0
path = 'images'
for file_name in os.listdir(path):
    cnt = cnt + 1
    if cnt % 10 == 0:
        print(file_name)
    else:
        os.remove(path + '/' + str(file_name))
        continue

