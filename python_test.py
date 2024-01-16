#EE596
#Lab 01 - Huffman Code
#ChiranjithD
#Original

import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import math

image = img.imread('Pattern.jpg')
#print(image.max)
plt.figure(1)
imgplot = plt.imshow(image)
plt.title("Original Parrots Image - E/17/047")
###############################################

#Step 04
#quantized_img = np.floor_divide(img_cropped, 256//8)
#quantized_img=np.around(img_cropped / (256/8-1)).astype(int)
quantized_img = np.digitize(image, np.arange(0, 256, 256/8)) - 1  #quantized the image


#plot the quantized image
plt.figure(2)
plt.title("Quantized Image(Original) - E/17/047")
quantized_img_32=quantized_img*(256//8) 
imgplot = plt.imshow(quantized_img_32)

#convert to 1D 
quantized_img_1d = quantized_img.reshape((1, -1))[0]
f = open("quantized_img_1d_original.txt", "a")
f.write(np.array2string(quantized_img_1d))
f.close()

########################################################
#Step 05
#hist_data = np.histogram(quantized_img_1d, bins=8)
#hist, _ = np.histogram(quantized_img_1d, bins=8, range=(0, 7), density=True)
#hist = hist / len(quantized_img_1d)
# Display the histogram and probabilities
hist, bins = np.histogram(quantized_img_1d, bins=8, range=(0, 7))
prob_dist = hist / np.sum(hist)

plt.figure(4)
plt.title("Probability Distribustion with Symbols - E/17/047")
symbols = np.arange(8)
plt.bar(symbols, hist / np.sum(hist))
plt.xticks(symbols)
plt.xlabel('Symbol')
plt.ylabel('Probability')
'''
print('Symbol probability distribution:')
for i, p in enumerate(hist):
    print(f'Symbol {i}: {p:.3f}')
'''
sorted_symbols = sorted(symbols, key=lambda s: hist[s], reverse=True)
probabilities=sorted(prob_dist,reverse=True)
print("symbols: ", sorted_symbols)
print("probabilities: ", probabilities)

############################################################
#Step 06
def build_huffman_tree(symbols, freq):
    # Initialize list of trees as leaf nodes
    trees = [[freq[i], [symbols[i], '']] for i in range(len(symbols))]
    
    while len(trees) > 1:
        # Sort the trees by frequency
        trees = sorted(trees, key=lambda x: x[0])
        #print("Tree",trees)
        #print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        # Take the two trees with lowest frequency
        left_tree = trees[0]
        right_tree = trees[1]
        # Add '0' to codewords of left tree and '1' to codewords of right tree
        for pair in left_tree[1:]:
            pair[1] = '1' + pair[1]
        for pair in right_tree[1:]:
            pair[1] = '0' + pair[1]
        # Create a parent node and add the frequency of children to it
        parent = [left_tree[0] + right_tree[0]] + left_tree[1:] + right_tree[1:]
        # Remove the children from the list and add the parent node
        trees = trees[2:]
        trees.append(parent)
    # Return the root node of the Huffman tree
    return trees[0]

# Build the Huffman tree for the symbols and their frequencies
huffman_tree = build_huffman_tree(sorted_symbols, probabilities)
codewords = dict(huffman_tree[1:])
print(codewords)

############################################################
#Step 07
encoding_output = []
for c in quantized_img_1d:
    #print(c,codewords[c])
    encoding_output.append(codewords[c])
Original_compressed_image = ''.join([str(item) for item in encoding_output]) 

############################################################
#Step 08
# write the compressed data string to a text file
with open('Original_compressed_image.txt', 'w') as file:
    file.write(Original_compressed_image)
#print(Cropped_compressed_image)

##################################################################
#Step 10
with open('Original_compressed_image.txt', 'r') as file:
    compressed_data = np.array([(num) for num in file.readline()])

def keyFind(value):
    key = None
    for k, v in codewords.items():
        if v == value:
            key = k
            return key
    
# decode the compressed data
decoded_data = []
current_code = ''
#decoded_image = np.array([codewords[symbol] for symbol in compressed_data])
codewords_values = codewords.values()
for bit in compressed_data:
    current_code += str(bit)
    current_code1="'"+current_code+"'"
    if current_code in codewords.values():
        key=keyFind(current_code)
        #print(key)
        decoded_data.append(key)
        current_code = ""

#print(decoded_data)
#print(quantized_img_1d)
decoded_pixels = np.array(decoded_data).reshape((680, 680, 3))
#print(decoded_pixels)

plt.figure(5)
plt.title(" Decompress Image(Original) - E/17/047")
decode_img_32=decoded_pixels*(256//8) 
imgplot = plt.imshow(decode_img_32)
plt.savefig('Decompress Image Original.jpg')
print(codewords)

###############################
#Step 11
entropy=0
for prob in probabilities:
    if prob > 0:
        entropy += prob * math.log2(1/prob)
print("Original Image Entropy: ",entropy)

#Step 12
psnro = peak_signal_noise_ratio(image, decoded_pixels)
print(f"The PSNR of Original Image: " ,psnro)


compressed_file_size = len(Original_compressed_image) 
compression_ratio = 680*680*8 / (compressed_file_size)
print(compression_ratio)


plt.show()