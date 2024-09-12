#Template_creation
from captcha import Captcha
import numpy as np, pathlib
import json, glob, cv2
import  os.path as osp, argparse
if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/wootc/Downloads/sampleCaptchas',
                        help='Path to the folder containing the input and output folders')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    args = parser.parse_args()

    filenames = glob.glob(osp.join(args.input_path,'input/*.jpg'))
    save_folder = './templates'
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    char_to_filenames = {} # character to filenames mapping
    filename_to_char = {} # filename to character mapping
    for filename in filenames:
        # read the output file
        output_filename = filename.replace('input', 'output').replace('.jpg', '.txt')
        if not osp.exists(output_filename):
            print(f'File {output_filename} does not exist')
            continue
        with open(output_filename, 'r') as f:
            txt = f.read().rstrip()
        for letter in txt:
            if letter not in char_to_filenames: char_to_filenames[letter] = []
            char_to_filenames[letter].append(filename)
        filename_to_char[filename] = txt

    letters = list(char_to_filenames.keys())
    #sort the letters by the number of images
    indices_sort_by_count = np.argsort([len(char_to_filenames[letter]) for letter in letters])
    np.random.seed(args.seed)
        
    to_be_collected = {key:None for key in letters}
    training_filenames = []
    #start with the least common letters
    for idx in indices_sort_by_count:
        key = letters[idx]
        if key not in to_be_collected: 
            continue
        #select a random image for the letter
        selected_filename = np.random.choice(char_to_filenames[key], 1, replace=False)[0]
        training_filenames.append(selected_filename)
        #remove the letter contained in this image from the to_be_collected list
        all_letters = filename_to_char[selected_filename]
        for character in all_letters:
            if character not in to_be_collected: 
                continue
            to_be_collected.pop(character)
    
    print("# of training samples: ", len(training_filenames))
    #save the templates
    for filename in training_filenames:
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im = Captcha.otsu_thresholding(im[Captcha.top:Captcha.bottom, Captcha.left:Captcha.right])
        letter_images = Captcha.crop_letters(im)
        letters = filename_to_char[filename]
        for idx, letter_image in enumerate(letter_images):
            cv2.imwrite(osp.join(save_folder, f'{letters[idx]}.png'), letter_image)
    
    #save the train-test split
    testing_filenames = list(set(filenames) - set(training_filenames))
    with open(osp.join(save_folder, 'train_test_split.json'), 'w') as f:
        json.dump({'train':training_filenames, 'test':testing_filenames},
                   f, indent=2)