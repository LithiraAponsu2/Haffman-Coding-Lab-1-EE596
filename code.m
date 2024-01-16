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
xPosition = 0;  % x position = 0*60 = 0
yPosition = 100; % y position = 25*4 = 100
width = 16;
height = 16;
CroppedImage = imcrop(OriginalImage, [xPosition yPosition width height-1]); % [xmin ymin w h]
figure(2);
imshow(CroppedImage);
title('Cropped Image E/18/025');

%Convert the original image into a grayscale image --------------
GrayImage=rgb2gray(OriginalImage);
figure(3);
imshow(GrayImage);
title('Gray Image E/18/025');

%Convert the cropped image into a grayscale image
GrayCropped=rgb2gray(CroppedImage)
figure(4);
imshow(GrayCropped);
title('Cropped Gray Image E/18/025');

% Step 4: Quantize the output at Step 3 into 8 levels (level 0-7) using uniform quantization.
% Calculate the maximum and minimum pixel values in the cropped grayscale image
minVal = 0; % min(GrayCropped(:));
maxVal = 255; % max(GrayCropped(:));

% Calculate the range of intensities
intensityRange = maxVal - minVal

% Define the number of quantization levels
numLevels = 8;

% Calculate the step size for quantization
stepSize = intensityRange / numLevels

% Perform uniform quantization manually
quantizedCropped = uint8(floor((double(GrayCropped) - minVal) / stepSize)*stepSize)  % double use to convert from uint8 to double

% Display the quantized image
figure(5);
imshow(quantizedCropped);
title('Quantized Cropped Image E/18/025');

% ================================================================================
% Step 5: Find the probability of each symbol distribution of the output at Step 4.
% Get unique symbols from the quantized image
uniqueSymbols = unique(quantizedCropped);

% Calculate the total number of pixels in the quantized image
totalPixels = numel(quantizedCropped);

% Initialize arrays to hold symbol frequencies and probabilities
numUniqueSymbols = numel(uniqueSymbols);
symbolFrequency = zeros(1, numUniqueSymbols);
symbolProbability = zeros(1, numUniqueSymbols);

% Calculate symbol frequency for each unique symbol
for i = 1:numUniqueSymbols
    symbolFrequency(i) = sum(quantizedCropped(:) == uniqueSymbols(i));
end

% Calculate the probability of each symbol
symbolProbability = symbolFrequency / totalPixels;

% Create a table for symbol frequency and probability
symbolTable = table(uniqueSymbols, symbolFrequency', symbolProbability','VariableNames', {'Symbol', 'Frequency', 'Probability'});
symbolTableSorted = sortrows(symbolTable,{'Probability'},'descend');

% Display the table
disp(symbolTableSorted);

x1=symbolTable.Symbol;
y1=symbolTable.Frequency;
figure(6);
bar(x1,y1);
title('Histogram E/18/025','FontSize',14);

% ===================================================================================
% Step 6: Construct the Huffman coding algorithm for cropped image at Step 4

huffmanDict = buildHuffmanTree(symbolTableSorted.Symbol, symbolTableSorted.Frequency);
% Display the Huffman dictionary
keys = huffmanDict.keys;
values = huffmanDict.values;
for i = 1:numel(keys)
    disp(['Symbol: ' keys{i} ', Code: ' values{i}]);
end


function huffmanDict = buildHuffmanTree(symbols, freq)
    % Initialize list of trees as leaf nodes
    trees = cell(length(symbols), 1);
    for i = 1:length(symbols)
        trees{i} = {freq(i), {symbols(i), ''}};
    end
    
    while length(trees) > 1
        % Sort the trees by frequency
        [~, idx] = sort(cellfun(@(x) x{1}, trees));
        trees = trees(idx);
        
        % Take the two trees with the lowest frequency
        leftTree = trees{1};
        rightTree = trees{2};
        
        % Add '0' to codewords of left tree and '1' to codewords of right tree
        leftTree{2}{2} = ['1' leftTree{2}{2}];
        rightTree{2}{2} = ['0' rightTree{2}{2}];
        
        % Create a parent node and add the frequency of children to it
        parent = {leftTree{1} + rightTree{1}, leftTree{2:end}, rightTree{2:end}};
        
        % Remove the children from the list and add the parent node
        trees = trees(3:end);
        trees{1} = parent;
    end
    
    % Return the Huffman dictionary
    huffmanDict = containers.Map;
    for i = 1:length(trees{1}) - 1
        key = trees{1}{i}{1};
        value = trees{1}{i}{2};
        huffmanDict(key) = value{2};
    end
end

% Assuming you have sorted_symbols and probabilities
% sortedSymbols = symbolTableSorted.Symbol; % Replace with your symbols
% probabilities = symbolTableSorted.Frequency; % Replace with your probabilities

% Build the Huffman tree for the symbols and their frequencies





% left = zeros(1, (numel(symbolTableSorted.Frequency) * 2 - 2)/2);  % store right node values
% right = zeros(1, (numel(symbolTableSorted.Frequency) * 2 - 2)/2);  % store left node values
% node = symbolTableSorted.Frequency; % temp for store nodes
% 
% 
% for i = 1:(numel(symbolTableSorted.Frequency) * 2 - 2) / 2
%     [min1, idx1] = min(node);  % Find the minimum value and its index
%     node(idx1) = Inf;  % Set the minimum value to Inf to find the next minimum
%     
%     [min2, idx2] = min(node);  % Find the second minimum value and its index
%     node(idx2) = Inf;  % Set the second minimum value to Inf
%     
%     left(i) = min1;  % Assign the minimum value to the left array
%     right(i) = min2;  % Assign the second minimum value to the right array
%     
%     node(end+1) = min1 + min2;  % Append the sum of the two minimum values to the node array
% end
% 
% symbolSorted = symbolTableSorted.Symbol';
% freqSorted = symbolTableSorted.Frequency';  % store freq for backtrace
% 
% % Initialize symbol strings
% symbolStrings = cell(size(symbolSorted));
% for i = 1:numel(symbolStrings)
%     symbolStrings{i} = ''; % Initialize each symbol string as an empty string
% end
% 
% % Huffman coding process
% for i = numel(right):-1:1
%     % Check if the value at the right array exists in freqSorted
%     idx = find(freqSorted == right(i), 1);
%     
%     % If found in freqSorted, assign '1' to the corresponding symbol in symbolStrings
%     if ~isempty(idx)
%         symbolStrings{idx} = '1';
%         freqSorted(idx) = []; % Remove the used frequency from the list
%         symbolSorted(idx) = []; % Remove the used symbol from the list
%         continue; % Move to the next iteration
%     end
%     
%     % If the value is not found in freqSorted, check the corresponding value in the left array
%     idx = find(freqSorted == left(i), 1);
%     
%     % If found in freqSorted, assign '0' to the corresponding symbol in symbolStrings
%     if ~isempty(idx)
%         symbolStrings{idx} = '0';
%         freqSorted(idx) = []; % Remove the used frequency from the list
%         symbolSorted(idx) = []; % Remove the used symbol from the list
%     end
% end
% 
% % Concatenate '0's to all symbols that have not been encoded yet
% for i = 1:numel(symbolStrings)
%     if isempty(symbolStrings{i})
%         symbolStrings{i} = '0';
%     end
% end
% 
% % Display the symbol strings
% for i = 1:numel(symbolStrings)
%     fprintf('Symbol %d: Huffman Code %s\n', symbolSorted(i), symbolStrings{i});
% end
% 
% 

% % =============================================================
% % Step 7: Compress both cropped and original images using the algorithm and the codebook
% % generated at step 6. You may round any intensity values outside the codebook, to the nearest
% % intensity value in the codebook, where necessary.


% cropped image
% symbolSorted = symbolTableSorted.Symbol; % Replace with your symbol array
% huffmanCode = ["1","01","0011","0010","0001","00001","00000"]; % Replace with your Huffman code array
% 
% % Raster scan
% % Transpose the cropped image to read row-wise
% quantizedTranspose = quantizedCropped.';
% 
% % Convert the transposed quantized image to a linear array (row-wise)
% quantizedLinearCrop = quantizedTranspose(:);
% 
% % Create a map to store Huffman codes for each symbol
% huffmanMapCrop = containers.Map(symbolSorted, huffmanCode);
% 
% % Initialize the encoded data for cropped image
% encodedDataCrop = '';
% 
% % Encode pixel intensities using the Huffman map
% for i = 1:numel(quantizedLinearCrop)
%     intensity = quantizedLinearCrop(i);
%     encodedDataCrop = strcat(encodedDataCrop, huffmanMapCrop(intensity)); % Append the corresponding Huffman code
% end
% 
% % original Image
% % Perform uniform quantization manually
% quantizedOriginal = uint8(floor((double(GrayImage) - minVal) / stepSize) * 31.875);
% figure(7);
% imshow(quantizedOriginal);
% title('Quantized original Image E/18/025');
% 
% % Quantize values that are outside the codebook range to the nearest available symbol
% quantizedLinearOrig = quantizedOriginal(:);
% quantizedLinearOrig(quantizedLinearOrig < min(symbolSorted)) = min(symbolSorted);
% quantizedLinearOrig(quantizedLinearOrig > max(symbolSorted)) = max(symbolSorted);
% 
% % Initialize the encoded data for original image
% encodedDataOrig = '';
% 
% % Encode pixel intensities using the Huffman map
% for i = 1:numel(quantizedLinearOrig)
%     disp(i);
%     intensity = quantizedLinearOrig(i);
%     encodedDataOrig = strcat(encodedDataOrig, huffmanMapCrop(intensity)); % Append the corresponding Huffman code
% end
% 
% % Step 8: Save the compressed image into a text file.
% % Save compressed cropped image data to a text file
% fileID = fopen('compressed_cropped_image.txt', 'wt');
% fprintf(fileID, '%s', encodedDataCrop);
% fclose(fileID);
% 
% % Save compressed original image data to a text file
% fileID = fopen('compressed_original_image.txt', 'wt');
% fprintf(fileID, '%s', encodedDataOrig);
% fclose(fileID);
% 
% % Step 9: Compress the original image using Huffman encoding function in the Matlab toolbox and
% % save it into another text file.
% % (Assuming the 'quantizedOriginal' variable is already generated and 'huffmanDict' is available from previous code)
% 
% % Convert the quantized original image to a linear array
% quantizedLinear = quantizedOriginal(:);
% 
% % Get unique symbols and their frequencies
% uniqueSymbols = unique(quantizedLinear);
% symbolFrequency = histc(quantizedLinear, uniqueSymbols);
% 
% % Calculate probabilities
% totalPixels = numel(quantizedLinear);
% symbolProbability = symbolFrequency / totalPixels;
% 
% % Generate Huffman dictionary
% huffmanDict = huffmandict(uniqueSymbols, symbolProbability);
% 
% % Encode pixel intensities using the Huffman dictionary
% encodedDataMatlabHuffman = huffmanenco(quantizedLinear, huffmanDict);
% 
% % Save the encoded data to a text file
% fileID = fopen('compressed_original_image_matlab.txt', 'wt');
% fprintf(fileID, '%d', encodedDataMatlabHuffman);
% fclose(fileID);
% 
% 
% % 10
% 
% %Decode the image encoded without using toolbox
% FileID=fopen('compressed_original_image_matlab.txt', 'r');
% EncodedStream=fscanf(FileID,'%s');
% EncodedImage1=zeros(1,length(EncodedStream));
% for i=1:length(EncodedStream)
%     EncodedImage1(i)=str2double(EncodedStream(i));
% end
% fclose(FileID);
% %Perform Huffman decoding
% DecodedImage1=huffmandeco(EncodedImage1, huffmanDict);
% DecodedImage1=DecodedImage1(1:612*612);
% %Reshape the decoded image back into the original shape
% OutputImage1=reshape(DecodedImage1,size(GrayImage));
% %Display the decompressed image
% figure(8);
% imshow(OutputImage1);
% title('Image encoded without toolbox E/18/025');
% 
% %Decode the image encoded with the toolbox
% %Read the encoded image from the text file
% FileID=fopen('compressed_original_image_matlab.txt', 'r');
% EncodedImage2=fscanf(FileID, '%d');
% fclose(FileID);
% %Perform Huffman decoding
% DecodedImage2=huffmandeco(EncodedImage2, huffmanDict);
% %Reshape the decoded image back into the original shape
% OutputImage2=reshape(DecodedImage2,size(GrayImage));
% %Display the decompressed image
% figure(9);
% imshow(OutputImage2);
% title('Image encoded with toolbox E/18/025');
% 
% %Step 11: Calculate the entropy of the Source
% %Entropy of the original Image
% EntropyOriginal=entropy(OriginalImage);
% disp("Entropy of the Original Image = ");
% disp(EntropyOriginal);
% 
% %Entropy of the Cropped Image
% EntropyCropped=entropy(CroppedImage);
% disp("Entropy of the Cropped Image = ");
% disp(EntropyCropped);
% 
% %Entropy of the Gray Image
% EntropyGray=entropy(GrayImage);
% disp("Entropy of the Gray Image = ");
% disp(EntropyGray);
% 
% %Entropy of the Cropped Gray Image
% EntropyGrayCropped=entropy(GrayCropped);
% disp("Entropy of the Cropped Gray Image = ");
% disp(EntropyGrayCropped);
% 
% %Step 12: Evaluate the PSNR of original images and decompressed images
% %PSNR of the decoded image which is encoded without using toolbox
% PSNRValue1 = psnr(OutputImage1, GrayImage);
% disp("PSNR of the decoded image which is encoded without using toolbox = ");
% disp(PSNRValue1);
% 
% %PSNR of the decoded image which is encoded with using toolbox
% PSNRValue2 = psnr(OutputImage2, GrayImage);
% disp("PSNR of the decoded image which is encoded with using toolbox = ");
% disp(PSNRValue2);




