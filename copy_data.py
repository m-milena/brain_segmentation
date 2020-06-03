import os
import shutil

def main():
    origin_path = './train/'
    dest_path = './dataset_ax1/train/'
    
    list_of_folders = [f for f in os.listdir(origin_path)]
    print(len(list_of_folders))

    sample_nr = 0
    for folder in list_of_folders:
        img_path = origin_path + folder +'/ax1/images/'
        label_path = origin_path + folder +'/ax1/labels/'
        files = [f for f in os.listdir(img_path)]

        for f in files:
            original = img_path + f
            target = dest_path + 'images/sample_%05d.png'%sample_nr
            shutil.copyfile(original, target)
            original = label_path + f
            target = dest_path + 'labels/sample_%05d.png'%sample_nr
            shutil.copyfile(original, target)
            sample_nr += 1
        print('Copied %s'%folder)





if __name__ == '__main__':
    main()
