import hashlib
import sys

if __name__ == '__main__':
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(sys.argv[1], 'rb') as f:
        b = f.read()
        md5.update(b)
        sha1.update(b)

    print('md5: {}\nsha1: {}'.format(md5.hexdigest(), sha1.hexdigest()))
