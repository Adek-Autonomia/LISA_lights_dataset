import numpy as np
from PIL import Image
import pandas as pd 
import os


home_dir = os.getcwd()
subdir_prefix = 'archive/dayTrain/dayTrain/dayClip'
annotation_prefix = 'archive/Annotations/Annotations/dayTrain/dayClip'
subdir_range = range(1, 14)
new_folder_loc1 = home_dir + 'processed/'
new_folder_loc2 = new_folder_loc1 + 'day'


os.chdir(home_dir)

cnt = 0
green_cnt = 0
red_cnt = 0

for s_num in subdir_range:
    subdir = subdir_prefix + str(s_num) + '/frames'
    anno_dir = annotation_prefix + str(s_num)
    os.chdir(home_dir)
    os.chdir(anno_dir)
    anno_file = pd.read_csv('frameAnnotationsBOX.csv', sep=';')
    os.chdir(home_dir)
    os.chdir(subdir)
    for root, dirs, files in os.walk('.'):
        allfiles = files
        break
    imgs = [ f for f in allfiles if f.split('.')[-1]=='jpg']
    for index, row in anno_file.iterrows():
        i = Image.open(row['Filename'].split('/')[-1])
        i_data = row
        imgarr = np.array(i)
        coords = [  [int(i_data['Upper left corner Y']),
                     int(i_data['Upper left corner X'])],
                    [int(i_data['Lower right corner Y']),
                     int(i_data['Lower right corner X'])]]
        # if coords[0][0] - coords[1][0] > -40 or coords[0][1] - coords[1][1] > -40:
        #     continue



        if i_data['Annotation tag'] == 'go':
            green_cnt += 1
            newfilename = 'green' + '_' + str(green_cnt) + '.jpg'
        elif i_data['Annotation tag'] == 'stop':
            red_cnt += 1
            newfilename = 'red' + '_' + str(red_cnt) + '.jpg'
        else:
            continue
        
        newimg = imgarr[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1], :]
        try:
            newi = Image.fromarray(newimg)
        except ValueError:
            continue
        os.chdir(new_folder_loc2)
        newi.save(newfilename)
        os.chdir(home_dir)
        os.chdir(subdir)

        cnt += 1
        if cnt%200==0:
            print(cnt, 'images done')
    print('Completed', s_num)






print('All done, total:', cnt)
print('Green:', green_cnt)
print('Red:', red_cnt)



















