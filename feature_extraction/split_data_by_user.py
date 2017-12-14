
import os
from shutil import copyfile








TRAIN_PATH = '../input/train/audio'
SPLIT_RATIO = 0.2
SPLIT_TRAIN_PATH = '../input/split_train/audio'
SPLIT_VAL_PATH = '../input/split_validation/audio'


global_train_set = set()
global_val_set = set()

def split_dir_by_user (dir_path):
	train_list = []
	val_list = []
	to_split = []
	for subdir, dirs, files in os.walk(dir_path):
		local_train_set = set()
		local_val_set = set()
		print ('Size of ' + dir_path + ': ' + str(len(files)))
		folder_size = len(files)
		for file in files:
			user_id = file.split('_')[0]
			#print(user_id)
			if user_id in global_train_set:
				train_list.append(file)
				local_train_set.add(user_id)
			elif user_id in global_val_set:
				val_list.append(file)
				local_val_set.add(user_id)
			else:
				to_split.append(file)
		max_val_list = int (SPLIT_RATIO * folder_size)

		for file in to_split:
			user_id = file.split('_')[0]
			if user_id in local_train_set:
				train_list.append(file)
			elif user_id in local_val_set:
				val_list.append(file)
			else:
				if len(val_list) < max_val_list:
					val_list.append(file)
					local_val_set.add(user_id)
					global_val_set.add(user_id)
				else:
					train_list.append(file)
					local_train_set.add(user_id)
					global_train_set.add(user_id)

		print ('--- SUMMARY OF ' + dir_path + ' ---')
		print ('TRAIN SET SIZE: ' + str(len(train_list)))
		print ('VALIDATION SET SIZE: ' + str(len(val_list)))
		print ('REAL SPLIT RATIO ' + str(float(len(val_list))/float(folder_size)))
		final_train_path = os.path.join(SPLIT_TRAIN_PATH, dir_path.split('/')[-1]) + '/'
		final_val_path = os.path.join(SPLIT_VAL_PATH, dir_path.split('/')[-1]) + '/'
		print (final_train_path)
		if not os.path.exists(os.path.dirname(final_train_path)):
			os.makedirs(os.path.dirname(final_train_path))
		if not os.path.exists(os.path.dirname(final_val_path)):
			os.makedirs(os.path.dirname(final_val_path))
		for file in train_list:
			copyfile(os.path.join(dir_path, file), os.path.join(final_train_path, file))
		for file in val_list:
			copyfile(os.path.join(dir_path, file), os.path.join(final_val_path, file))







subdirs = os.listdir(TRAIN_PATH)
for subdir in subdirs:
	split_dir_by_user(os.path.join(TRAIN_PATH,subdir)) 
			