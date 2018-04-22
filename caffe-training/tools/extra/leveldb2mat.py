import leveldb
import feat_helper_pb2
import numpy as np
import scipy.io as sio
import time

def main(argv):
	leveldb_name = sys.argv[1]
	print "%s" % sys.argv[1]
	# print "%s" % sys.argv[2]
	# print "%s" % sys.argv[3]
	# print "%s" % sys.argv[4]
        # window_num = 1000;
        # window_num = 12736;
        # window_num = int(sys.argv[2]);
        # window_num = 2845;

	start = time.time()
	if 'db' not in locals().keys():
		db = leveldb.LevelDB(leveldb_name)
		datum = feat_helper_pb2.Datum()

	
	# iter = db.RangeIter(key_from ="%08d" %(0), include_value = False)
	iter = db.RangeIter(include_value = False)
	
	window_num = len(list(iter))
	print window_num
	# while (iter = iter.next())

	datum.ParseFromString(db.Get("%08d" %(0)))
	dim = len(datum.float_data)
	print dim
	ft = np.zeros((window_num, dim),dtype=np.float32)
	for im_idx in range(window_num):
		datum.ParseFromString(db.Get("%08d" %(im_idx)))
		ft[im_idx, :] = datum.float_data

	# print 'time 1: %f' %(time.time() - start)
	sio.savemat(sys.argv[2], {'feats':ft},oned_as='row')
	print 'time 2: %f' %(time.time() - start)
	print 'done!'

	#leveldb.DestroyDB(leveldb_name)

if __name__ == '__main__':
	import sys
	main(sys.argv)
