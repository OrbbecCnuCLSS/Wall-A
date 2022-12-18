import os

cnt = 0
path = 'images/'
for file_name in os.listdir(path):
    print(file_name)
    os.rename(path + file_name, path + '{:0>3d}'.format(cnt) + '.png')
    cnt = cnt + 1    
