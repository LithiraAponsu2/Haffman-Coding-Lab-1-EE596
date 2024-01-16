% E/18/025
% APONSU G.L.C.
% Nov-2023

clc
clear all
close all

% Step 2: Read the original image into a Matrix.
% 25 < 150 -> selected Pattern.jpg
OriginalImage = imread('Pattern.jpg');
figure(1);
imshow(OriginalImage);
title('Original Image E/18/025');

% Step 3: Select 16×16 cropped sub-image from your input at step 2
% x position = 0*60 = 0
% y position = 25*4 = 100
% strarting point of crop window (0,100)
CroppedImage=imcrop(OriginalImage,[0 100 16 15]);  % [xmin ymin w h]
figure(2);
imshow(CroppedImage);
title('Cropped Image E/18/025','FontSize',14);


