import sys
import tarfile

tar = tarfile.open(sys.argv[1])
tar.extractall()
tar.close()
