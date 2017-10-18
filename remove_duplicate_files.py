import hashlib
import os
from time import clock as now


def getmd5(filename):
    file_txt = open(filename, 'rb').read()
    m = hashlib.md5()
    m.update(file_txt)
    return m.hexdigest()


def main():
    # path = raw_input("path: ")
    path = './'
    all_size = {}
    total_file = 0
    total_delete = 0
    start = now()
    for file in os.listdir(path):
        total_file += 1
        real_path = os.path.join(path, file)
        if os.path.isfile(real_path):
            size = os.stat(real_path).st_size
            name_and_md5 = [real_path, '']
            if size in all_size.keys():
                new_md5 = getmd5(real_path)
                if all_size[size][1] == '':
                    all_size[size][1] = getmd5(all_size[size][0])
                if new_md5 in all_size[size]:
                    os.remove(file)
                    print('DELETE', file)
                    total_delete += 1
                else:
                    all_size[size].append(new_md5)
            else:
                all_size[size] = name_and_md5
    end = now()
    time_last = end - start
    print('Total files:', total_file)
    print('Deleted files:', total_delete)
    print('Time using:', time_last, '秒')


if __name__ == '__main__':
    main()
