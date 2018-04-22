import leveldb
import lmdb
import feat_helper_pb2
import numpy as np
import scipy.io as sio
import time

def main(argv):
	leveldb_name = sys.argv[1]
	db_save_name = sys.argv[4]
	print "%s" % sys.argv[1]
	print "%s" % sys.argv[2]
	print "%s" % sys.argv[3]
	print "%s" % sys.argv[4]
	print "%s" % sys.argv[5]
	temp = int(sys.argv[5])
        # window_num = 1000;
        # window_num = 12736;
	window_num = int(sys.argv[2]);
        # window_num = 2845;

	start = time.time()
	if 'db' not in locals().keys():
		db = leveldb.LevelDB(leveldb_name)		
		datum = feat_helper_pb2.Datum()
	db_save = lmdb.open(db_save_name, map_size=int(1e12))
	#batch = leveldb.WriteBatch()
	datum_save = feat_helper_pb2.Datum()
	#ft = np.zeros((window_num, int(sys.argv[3])))
	for im_idx in range(window_num):
		
		key_str = '%08d' % (im_idx)
		datum.ParseFromString(db.Get(key_str))
		ft0 = np.array(datum.float_data)
		#print ft0.shape
		ft1 = ft0 / temp
		m_ft1 = ft1.max()
		ft1 = ft1 - m_ft1
		exp_ft1 = np.exp(ft1)
		exp_ft1_s = exp_ft1.sum()
		p_ft1 = exp_ft1 / exp_ft1_s
		del datum_save.float_data[:]
		datum_save.float_data.extend(p_ft1.tolist())
		#batch.Put(key_str, datum_save.SerializeToString())
		db_save.begin(write=True).put(key_str,datum_save.SerializeToString())
		if (im_idx % 10000 ==0):
			print "%d images processed." %(im_idx)
		# if (im_idx % 500 == 0 ):
		# 	#print np.array(datum.float_data)
		# 	print "sample %d" %(im_idx)
		# 	print p_ft1.max()
		# 	#calculate p when t=1
			
		# 	m_ft0 = ft0.max()
		# 	ft0 = ft0 - m_ft0
		# 	exp_ft0 = np.exp(ft0)
		# 	exp_ft0_s = exp_ft0.sum()
		# 	p_ft0 = exp_ft0 / exp_ft0_s

		# 	print p_ft0.max()



	#write leveldb
	#db_save.Write(batch, sync = True)
	db_save.close()
	print 'temperature is %d' %(temp)
	print 'time 1: %f' %(time.time() - start)
	#sio.savemat(sys.argv[4], {'feats':ft},oned_as='row')
	print 'time 2: %f' %(time.time() - start)
	print 'done!'

	#leveldb.DestroyDB(leveldb_name)

if __name__ == '__main__':
	import sys
	main(sys.argv)
