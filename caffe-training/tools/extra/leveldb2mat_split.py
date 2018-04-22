import leveldb
import feat_helper_pb2
import numpy as np
import scipy.io as sio
import time
def main(argv):
	leveldb_name = sys.argv[1]
	print "%s" % sys.argv[1]
	print "%s" % sys.argv[2]
	
        # window_num = 1000;
        # window_num = 12736;
	
        # window_num = 2845;
	iter = db.RangeIter(include_value = False)
	
	window_num = len(list(iter))
	print window_num
	# while (iter = iter.next())

	datum.ParseFromString(db.Get("%08d" %(0)))
	dim = len(datum.float_data)
	print dim
	
	if 'db' not in locals().keys():
		db = leveldb.LevelDB(leveldb_name)
		datum = feat_helper_pb2.Datum()

	v_per_split = 1e8
	split = np.ceil(float(window_num) * dim / v_per_split)
	per_batch = int(np.ceil(float(window_num) / split))
	for i in range(split):
		start = i*per_batch
		final = min((i+1)*per_batch,window_num)
		ft = np.zeros((final-start, dim),dtype=np.float32)
		for im_idx in range(start,final):
			datum.ParseFromString(db.Get("%08d" %(im_idx)))
			ft[im_idx-start, :] = datum.float_data

		
		sio.savemat(sys.argv[2]+'_'+str(i+1), {'feats':ft},oned_as='row')

	
	print 'done!'

	#leveldb.DestroyDB(leveldb_name)

if __name__ == '__main__':
	import sys
	main(sys.argv)
