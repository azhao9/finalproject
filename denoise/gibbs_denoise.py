import random
import numpy as np
import sys
from functions.io_data import read_data, write_data
from PIL import Image

class IsingModel:
    def __init__(self, image, ext_factor, beta):

        self.width, self.height, self.ext_factor, self.beta = image.shape[0], image.shape[1], ext_factor, beta
        self.image = image

    def neighbours(self, x, y):
        n = []
        if x == 0:
            n.append((self.width-1, y))
        else:
            n.append((x-1, y))
        if x == self.width-1:
            n.append((0, y))
        else:
            n.append((x+1, y))
        if y == 0:
            n.append((x, self.height-1))
        else:
            n.append((x, y-1))
        if y == self.height-1:
            n.append((x, 0))
        else:
            n.append((x, y+1))
        return n

    def local_energy(self, x, y):
        return self.ext_factor[x,y] + sum(self.image[xx,yy] for (xx, yy) in self.neighbours(x, y))

    def gibbs_sample(self, x, y):
        p = 1 / (1 + np.exp(-2 * self.beta * self.local_energy(x,y)))
        if random.uniform(0, 1) <= p:
            self.image[x, y] = 1
        else:
            self.image[x, y] = -1


def denoise(image, q, burn_in, iterations):
    external_factor = 0.5 * np.log(q / (1-q))
    model = IsingModel(image, external_factor*image, 3)

    avg = np.zeros_like(image).astype(np.float64)
    for i in range(burn_in + iterations):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if(random.uniform(0, 1) <= 0.7):
                    model.gibbs_sample(x, y)
        if(i > burn_in):
            avg += model.image
    return avg / iterations

def read_image_and_binarize(image_file):
    im = Image.open(image_file).convert("L")
    A = np.asarray(im).astype(int)
    A.flags.writeable = True

    A[A<128] = -1
    A[A>=128] = 1
    return A

def convert_from_matrix_and_save(M, filename, display=False):
  M[M==-1] = 0
  M[M==1] = 255
  im = Image.fromarray(np.uint8(M))
  if display:
    im.show()
  im.save(filename)

def get_mismatched_percentage(orig_image, denoised_image):
    diff = abs(orig_image - denoised_image)/2
    return (100.0 * np.sum(diff)) / np.size(orig_image)

def main():

        # 0096670_RP-P-OB-8087.jpg

        noise_type = sys.argv[1]
        orig_image = read_image_and_binarize(sys.argv[2])

        fname = sys.argv[2].split('/')[-1]

        noisy_image = read_image_and_binarize('raw_noisy_imgs/'+noise_type+'/'+fname)

        denoised_image = denoise(noisy_image, 0.7, 5, 10)

        denoised_image[denoised_image >= 0] = 1
        denoised_image[denoised_image < 0] = -1

        #width = avg.shape[0]
        #height = avg.shape[1]
        #counter = 0

        #for i in range(0, width):
        #    for j in range(0, height):
        #        data[counter][2] = avg[i][j][0]
        #        counter = counter + 1

        #write_data(data, "output.txt")
        #read_data("output.txt", True, save=True, save_name="output.jpg")

        print(fname+',', get_mismatched_percentage(orig_image, denoised_image))

        convert_from_matrix_and_save(denoised_image, 'bin_denoised_imgs/gibbs/'+noise_type+'/'+fname, display=False)

if __name__ == "__main__":
    main()
