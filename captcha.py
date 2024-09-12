import cv2, numpy as np, glob
import os.path as osp, pathlib

class Captcha(object):
    left, top, right, bottom = 4, 10, 50, 22
    def __init__(self):
        self.templates = []
        self.labels = []    
        for filename in glob.glob('templates/*.png'):
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.templates.append(im)
            self.labels.append(osp.splitext(osp.basename(filename))[0])
        self.templates = np.dstack(self.templates)

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im = Captcha.otsu_thresholding(im[Captcha.top:Captcha.bottom, Captcha.left:Captcha.right])
        
        letters = Captcha.crop_letters(im)
        result = ''
        for letter in letters:
            index = np.argmin(np.sum((self.templates - letter[..., np.newaxis])**2, axis=(0,1)))
            label = self.labels[index]
            result += label

        pathlib.Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(result)
        return result
        
    def otsu_thresholding(im_gray):
        """
        Applies Otsu's thresholding algorithm to convert a grayscale image to binary.
        Parameters:
            im_gray (numpy.ndarray): Grayscale image to be thresholded.
        Returns:
            numpy.ndarray: Binary image after applying Otsu's thresholding.
        """
        _, im_bw = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return im_bw
    
    def crop_letters(im): 
        """
        Crop the input image into individual letters.

        Parameters:
        - im: numpy.ndarray
            The input image.

        Returns:
        - letters: list
            A list of cropped letter images.
        """
        letters = []
        for idx in range(5):
            letters.append(im[:, idx*9:idx*9+10])
        return letters

if __name__=='__main__':
    import tqdm, matplotlib.pyplot as plt, argparse, json
    
    def update_performance(performance_dict, expected, output):
        performance_dict['sample_count'] += 1
        performance_dict['char_count'] += len(expected)
        if expected == output:
            performance_dict['correct_sample_count'] += 1
        for e, o in zip(expected, output):
            performance_dict['correct_char_count'] += e==o
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/wootc/Downloads/sampleCaptchas',
                        help='Path to the folder containing the input and output folders')
    args = parser.parse_args()
    filenames = glob.glob(osp.join(args.input_path,'input/*.jpg'))
    captcha = Captcha()
    visualize = False
    
    #read train test split
    with open('templates/train_test_split.json', 'r') as f:
        train_test_split = json.load(f)
    



    test_performance = {'sample_count':0, 'char_count': 0, 'correct_sample_count':0, 'correct_char_count':0}
    overall_performance = {'sample_count':0, 'char_count': 0, 'correct_sample_count':0, 'correct_char_count':0}
    for filename in tqdm.tqdm(filenames):
        output = captcha(filename, None)
        output_filename = filename.replace('input', 'output').replace('.jpg', '.txt')
        with open(output_filename, 'r') as f:
            expected = f.read().rstrip()
        if filename in train_test_split['test']:
            update_performance(test_performance, expected, output)
        update_performance(overall_performance, expected, output)
        if visualize:
            im_orig = cv2.imread(filename)
            title_string = f'Expected: {expected}, Output: {output}, Match: {expected==output}'
            plt.imshow(im_orig[..., ::-1])
            plt.title(title_string)
            plt.show()
        # break
    ## markdown performance table generation
    header = ["Dataset", "# of character", "Accuracy per character", "# of sample", "Accuracy per sample"]
    print("|".join(header))
    print("|".join(['---']*len(header)))
    print("|".join(['Overall', str(overall_performance['char_count']), 
                    f"{100*overall_performance['correct_char_count']/overall_performance['char_count']:.2f}%",
                    str(overall_performance['sample_count']),
                    f"{100*overall_performance['correct_sample_count']/overall_performance['sample_count']:.2f}%"]))
    print("|".join(['Test', str(test_performance['char_count']),   
                    f"{100*test_performance['correct_char_count']/test_performance['char_count']:.2f}%",
                    str(test_performance['sample_count']),
                    f"{100*test_performance['correct_sample_count']/test_performance['sample_count']:.2f}%"]))
