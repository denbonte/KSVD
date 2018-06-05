
% 22 MAR - v2.0
%   - function separation;
%   - codewords dropout in the KSVD_codebook_gen script (new).

% This code implements the K-SVD method starting from the original paper
% "K -SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse
% Representation", by Michal Aharon, Elad and Bruckstein.

% OBJECTIVE: basically the same of the simple VQ, that is to find a 
% dictionary that  can be used to reconstruct an arbitrary image with the 
% smallest possible error. Processing is made block-wise, where the blocks
% as usually are unwrapped and treated as vectors.

% DATA STRUCTURE NEEDED: 
%   1) N signals y_i of length "n", that can be  stored into/ seen as the 
%   matrix Y that has n rows and N columns, so that each column represents 
%   a y_i ("n" rows) and Y has N columns;
%   2) a codebook D that can be initialized with random values or with a
%   precise logic (i.e. constant values and edges, as seen in lab_10).
%   The codebook has K << N words, or codewords, each and every one of them 
%   of length equal to the length of the signal, i.e. "n". We can store the
%   codebook as a matrix "D", that will have n rows and K columns (each
%   column is a codeword);
%   3) a matrix X that stores the coefficients by which we must multiply
%   each codeword to obtain, as their linear combination, a signal that
%   approximates the one in the training set.

% ALGORITHM OUTLINE
% Main steps, as in K-Means, are:
%   1) initialization of the D matrix with L2 normalized columns, that is
%   each column must have L2 norm || . ||2 = 1;
%   2) Sparse coding stage - using an algorithm like matching pursuit,
%   compute the representation of each training set vector y_i in terms of
%   columns of D (codewords) such that it is sparse, that is the L0 norm of
%   the x column is bounded by a certain value T0. This problem can be
%   solved by exploiting one of the classical pursuit matching algorithms
%   such as MP (matching pursuit) or OMP (orthogonal matching pursuit) or
%   more sophisticated algorithms such as FOCUSS;
%   3) for every column of D (i.e. codeword) d_k, modify the codeword and 
%   the associated x's (that is coefficients used to weight d_k in those
%   combination that used the column to approximate some y_i) according to
%   an SVD decomposition, a generalization of the eigendecomposition for
%   non-squared matrices.

%   NOTE: here the codewords are filled with doubles and not with uint8 as
%   in VQ. Moreover, the training set must be generated synthetically 
%   (see the paper for further information)
%   Here we'll use directly an image to test the training.

%% CODEBOOK LOADING

clear;
close all;
clc;

codebooksPath  = 'learned_codebooks/';

[fileName, filePath] = uigetfile(fullfile(codebooksPath, '*.mat'), ...
    'Select learned codebook (.mat)'); 

temp = load(strcat(filePath, fileName));

D = temp.D;

[bdsqrd, K] = size(D);

bd = sqrt(bdsqrd);

% full test and comparison
fullTest = false;
testUpperLim = 20;
testLowerLim = 3;

if strcmp(input(sprintf('Run full test (T0 from %d to %d)? (yes/no): ', testLowerLim, testUpperLim), 's'), 'yes') == 1
   fullTest = true; 
end

%% CODEBOOK VISUALIZATION

codebookVisualiz = false;

if strcmp(input('Show the learned codebook (first 128 codewords)? (yes/no): ', 's'), 'yes') == 1
   codebookVisualiz = true; 
end

if codebookVisualiz == true
    learnedCodebook = cell(K, 1);

    for i = 1:K
        % reshape every (bd*bd)x1 vector into a (bd)x(bd) matrix, store it as a
        % cell in a cell matrix so that it can be shown with imshow
        learnedCodebook{i} = reshape(D(:, i), bd, bd);
    end

    % reshape (if necessary)
    % learnedCodebookMat = reshape(learnedCodebook, rows/bd, cols/bd);

    % visualize the first 128 codewords
    learnedCodebookFig = figure(2);
    set(learnedCodebookFig, 'name', 'codebook visualization', 'numbertitle', 'on');

    for i = 1:128
        subplot(8, 16, i);
        imshow(cell2mat(learnedCodebook(i)), []);
    end
end

%% DELETE NON-USED CODEWORDS

% codewordDropout = false;
% 
% if strcmp(input('Implement dropout of non used-codewords? (yes/no)', 's'), 'yes') == 1
%    codewordDropout = true; 
% end
% 
% % 
% % nonZeroCoeffs = [imgCoeffs ~= 0];
% % 
% % codewordUsage = sum(nonZeroCoeffs');
% % 
% % zeroUsage = find(~codewordUsage);
% % 
% % D(:, zeroUsage) = [];

%% CODEBOOK PRUNING 
% NOTE: could be implemented directly in "ksvd.m" to speed up the process.

codebookPruning = false;
pruningThresh   = 0.000005;

if fullTest == false && ...
        strcmp(input(sprintf('Apply pruning (threshold = %1.7f)? (yes/no): ', pruningThresh), 's'), 'yes') == 1
   codebookPruning = true; 
end

if codebookPruning == false && fullTest == false
   T0 = input('Number of K-SVD coeff.s to keep on single run (default is 10): ');

   if isempty(T0)
        T0 = 10;
   end
 
end

if codebookPruning == true
    for i = 1:size(D, 2)

        % avoid to repeat comparisons
        k = i + 1;

        % as long as we have codewords to compare
        while k < size(D, 2)

    %         if i == k
    %             k = k + 1;
    %             continue; 
    %         end

            blockMSE = sum(sum((D(:, i) - D(:, k)).^2))/(bd*bd);

            if blockMSE < 0.000005
                D(:, k) = [];
                k = k - 1;
            end

            k = k +1;

        end
    end

    % update codebook dimension
    K = size(D, 2);

    % recompute K to prevent particular cases in which K < T0
    T0 = min([K, T0]);
end

%% IMAGE LOADING AND PRE-ELAB
% (to separate function in which we load the learned codebook...)

% select the image to reconstruct with the learned codebook
% [fileName, filePath] = uigetfile(fullfile(basePath, '*.tiff'), ...
%     'Select instance input file (.tiff)'); 

testImagesPath = 'test_images/';

[fileName, filePath] = uigetfile(fullfile(testImagesPath, {'*.jpg; *.tiff'}), ...
    'Select instance input file (.jpg, .tiff)'); 

img = imread(strcat(filePath, fileName));

% size of the image
[rows, cols, chs] = size(img);

% if the image is encoded in RGB, we convert it in YCbCr and select only
% the luminance channel
if chs > 1
    img = myRGB2YCbCr(img);
end

% resize to fit blocks
img = imresize(img, [bd*round(rows/bd), bd*round(cols/bd)]);

[rows, cols] = size(img);

% resize the image (make it smaller in order to render the process 
% computationally feasible).. JUST FOR TEST REASON!
testResizeFactor = 1;
img = imresize(img, testResizeFactor);

rows = rows*testResizeFactor;
cols = cols*testResizeFactor;

% we compute the entropy to have an idea of how many bits it would really
% take, in average, to send a pixel of the original image
imgH = entropy(img);

% .. then we cast to double (entropy needs uint8)
img = double(img);
img = img-mean(mean(img));

% gaussian kernel definition
dim   = 3;
sigma = 0.5;
kernel   = fspecial('gaussian', dim, sigma);

if fullTest == true
   
    T0vec = testLowerLim:testUpperLim;

    % MSEs at different T0s
    mseVecKSVD     = zeros(size(T0vec)); 
    mseVecKSVDfilt = zeros(size(T0vec));
    mseVecDCT      = zeros(size(T0vec));
    
    % SNRs at different T0s
    snrVecKSVD     = zeros(size(T0vec)); 
    snrVecKSVDfilt = zeros(size(T0vec));
    snrVecDCT      = zeros(size(T0vec));
    
    % PSNRs at different T0s
    psnrVecKSVD     = zeros(size(T0vec)); 
    psnrVecKSVDfilt = zeros(size(T0vec));
    psnrVecDCT      = zeros(size(T0vec));
    
    % for index
    idx = 1;
   
    for T0 = T0vec
    
        % compute the KSVD transform saving T0 coefficients
        [~, recImg] = ksvd(img, D, bd, T0);

        % actual filtering
        filtd = conv2(recImg, kernel, 'same');
        
        % compute the DCT transform saving T0 coefficients
        recImgDCT = myDCT_v2(img, bd, T0);
        
        mseVecKSVD(idx)     = sum(sum((img - recImg).^2))/(rows*cols);
        mseVecKSVDfilt(idx) = sum(sum((img - filtd).^2))/(rows*cols);
        mseVecDCT(idx)      = sum(sum((img - recImgDCT).^2))/(rows*cols);

        
        psnrVecKSVD(idx)     = 10*log10(255^2/mseVecKSVD(idx));
        psnrVecKSVDfilt(idx) = 10*log10(255^2/mseVecKSVDfilt(idx));
        psnrVecDCT(idx)      = 10*log10(255^2/mseVecDCT(idx));
        
        [~,     snrVecKSVD(idx)]    = psnr(recImg, img);
        [~, snrVecKSVDfilt(idx)]    = psnr(filtd, img);
        [~,      snrVecDCT(idx)]    = psnr(recImgDCT, img);

        idx = idx + 1;
    end
else
    % compute the transform.s one time only
    [~, recImg] = ksvd(img, D, bd, T0);
    recImgDCT = myDCT_v2(img, bd, T0);
end

%% IMAGE FILTERING (KSVD) AND VISUALIZATION

% gaussian kernel definition
dim   = 3;
sigma = 0.5;
kernel   = fspecial('gaussian', dim, sigma);

% actual filtering
filtd = conv2(recImg, kernel, 'same');

% compute MSEs and PSNRs (ksvd, filt ksvd, dct)
mse     = sum(sum((img - recImg).^2))/(rows*cols);
mseFilt = sum(sum((img - filtd).^2))/(rows*cols);
mseDCT  = sum(sum((img - recImgDCT).^2))/(rows*cols);

peaksnr         = 10*log10(255^2/mse);
peaksnrFilt     = 10*log10(255^2/mseFilt);
peaksnrDCT      = 10*log10(255^2/mseDCT);

[~, snr]        = psnr(recImg, img);
[~, snrFilt]    = psnr(filtd, img);
[~, snrDCT]     = psnr(recImgDCT, img);

orksvdFig = figure(2);
set(orksvdFig, 'name', 'K-SVD reconstructed', 'numbertitle', 'on');
set(orksvdFig, 'units','normalized','outerposition', [0 0 1 1]);

subplot(2, 2, 1:2)
    imshow(img, []);
    title(sprintf('Original image "%s" ', fileName), 'interpreter', 'latex', 'fontsize', 13);
subplot(2, 2, 3)
    imshow(recImg, [])
    titLine1 = sprintf('Reconstructed image, %d coeff.s per %dx%d block', T0, bd, bd);
    titLine2 = sprintf('MSE = %g', mse);
    title(sprintf('\\begin{tabular}{c} %s %s %s \\end{tabular}', titLine1,'\\', titLine2),...
    'interpreter','latex', 'fontsize', 13)
subplot(2, 2, 4)
    imshow(filtd, [])
    titLine1 = sprintf('Reconstructed filtered image');
    titLine2 = sprintf('MSE = %g', mseFilt);
    title(sprintf('\\begin{tabular}{c} %s %s %s \\end{tabular}', titLine1,'\\', titLine2),...
    'interpreter','latex', 'fontsize', 13)


ksvddctFig = figure(3);
set(ksvddctFig, 'name', 'K-SVD vs DCT', 'numbertitle', 'on');
set(ksvddctFig, 'units','normalized','outerposition', [0 0 1 1]);

subplot(2, 2, 1:2)
    imshow(img, []);
    title(sprintf('Original image "%s" ', fileName), 'interpreter', 'latex', 'fontsize', 13);
subplot(2, 2, 3)
    imshow(recImg, [])
    titLine1 = sprintf('Reconstructed image, %d coeff.s per %dx%d block', T0, bd, bd);
    titLine2 = sprintf('MSE = %g', mse);
    title(sprintf('\\begin{tabular}{c} %s %s %s \\end{tabular}', titLine1,'\\', titLine2),...
    'interpreter','latex', 'fontsize', 13)
subplot(2, 2, 4)
    imshow(recImgDCT, [])
    titLine1 = sprintf('Reconstructed image (DCT), %d coeff.s per %dx%d block', T0, bd, bd);
    titLine2 = sprintf('MSE = %g', mseDCT);
    title(sprintf('\\begin{tabular}{c} %s %s %s \\end{tabular}', titLine1,'\\', titLine2),...
    'interpreter','latex', 'fontsize', 13)


%% MSE, SNR AND PSNR PLOTS

if fullTest == true
    
    mseComparisonFig = figure;
    set(mseComparisonFig, 'name', 'MSE comparison', 'numbertitle', 'on');
    set(mseComparisonFig, 'units','normalized','outerposition', [0 0 1 1]);
    plot(T0vec, mseVecKSVD); hold on;
    plot(T0vec, mseVecKSVDfilt);
    plot(T0vec, mseVecDCT);
    leg = legend('KSVD', '$filt_{N}$(KSVD)', 'DCT');
    set(leg, 'interpreter', 'latex', 'fontsize', 13, 'location', 'best');
    grid on;
    title(sprintf('Comparison of MSEs for KSVD, filtered KSVD and DCT for "%s"', fileName),...
        'interpreter', 'latex', 'fontsize', 13);
    xlabel('Number of coefficients kept', 'interpreter', 'latex', 'fontsize', 13);
    ylabel('MSE', 'interpreter', 'latex', 'fontsize', 13);
    
    snrComparisonFig = figure;
    set(snrComparisonFig, 'name', 'SNR comparison', 'numbertitle', 'on');
    set(snrComparisonFig, 'units','normalized','outerposition', [0 0 1 1]);
    plot(T0vec, snrVecKSVD); hold on;
    plot(T0vec, snrVecKSVDfilt);
    plot(T0vec, snrVecDCT);
    leg = legend('KSVD', '$filt_{N}$(KSVD)', 'DCT', 'location', 'best');
    set(leg, 'interpreter', 'latex', 'fontsize', 13);
    grid on;
    title(sprintf('Comparison of SNRs for KSVD, filtered KSVD and DCT for "%s"', fileName),...
        'interpreter', 'latex', 'fontsize', 13);    xlabel('Number of coefficients kept', 'interpreter', 'latex', 'fontsize', 13);
    ylabel('SNR', 'interpreter', 'latex', 'fontsize', 13);
    
    psnrComparisonFig = figure;
    set(psnrComparisonFig, 'name', 'PSNR comparison', 'numbertitle', 'on');
    set(psnrComparisonFig, 'units','normalized','outerposition', [0 0 1 1]);
    plot(T0vec, psnrVecKSVD); hold on;
    plot(T0vec, psnrVecKSVDfilt);
    plot(T0vec, psnrVecDCT);
    leg = legend('KSVD', '$filt_{N}$(KSVD)', 'DCT');
    set(leg, 'interpreter', 'latex', 'fontsize', 13, 'location', 'best');
    grid on;
    title(sprintf('Comparison of PSNRs for KSVD, filtered KSVD and DCT for "%s"', fileName),...
        'interpreter', 'latex', 'fontsize', 13);    xlabel('Number of coefficients kept', 'interpreter', 'latex', 'fontsize', 13);
    ylabel('PSNR [dB]', 'interpreter', 'latex', 'fontsize', 13);
    
end

%% INVESTIGATION ON CODING SCHEMES FOR THE COEFFICIENT MATRIX

% % minCoeff = min(min(imgCoeffs));
% % 
% % quantCoeffs = uencode(imgCoeffs(:),8);
% % 
% % 
% % %%
% % 
% % 
% % figure;
% % hist(imgCoeffs(:), 100);
% % axis tight;
