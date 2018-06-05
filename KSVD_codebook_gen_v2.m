
clear;
close all;
clc;


% INITIALIZE BLOCKS FROM TRAINING IMAGES

%% TRAINING SET LOADING 

basePath      = '';
trainingPath  = 'training/';

addpath(basePath);

% create a struct containing all the images in the directory
trainingFileList  = dir(strcat(trainingPath, '*'));

% delete "." and ".."
trainingFileList = trainingFileList(3:end);

% we load (at maximum) every image, convert them to YCbCr, and stitch all 
% the luminance component together.
% NOTE: there are about 20 images per person (frames from a video,
% probably...)

% data structure pre-allocation
trainingSet = cell(150, 1);

% block dimension
bd = 8;

fprintf('Fitting on dpt. of CV, University of Essex faces database.\n')
fprintf('Total number of images is 3040 on 152 subjects (20 images per subject).\n\n')

randomFaces = false;

if strcmp(input('Extract random samples from the training set? (yes/no): ', 's'), 'yes') == 1
   randomFaces = true; 
end

nImagesTrain = input('How many training samples? (min 1, max 3040, default is 10): ');

if isempty(nImagesTrain) || nImagesTrain < 1 || nImagesTrain > length(trainingFileList)
    nImagesTrain = 10;
end

if randomFaces == true
    trImagesList = randperm(length(trainingFileList), nImagesTrain);
else
    trImagesList = 1:20:nImagesTrain*20;
end

idx = 1;

for i = trImagesList
    
    % load the actual image
    imgRGB = imread(strcat(trainingPath, trainingFileList(i).name));
    
    % convert it into YCbCr and extract the luminance component
    imgY   = double(myRGB2YCbCr(imgRGB)); 
    
    % resize it so that it can be divided in 8x8 blocks
    imgY = imresize(imgY, [bd*round(size(imgY, 1)/bd), bd*round(size(imgY, 2)/bd)]);
    
    % stack images to form a big image using cells (vector form), removing
    % the mean from each image (so that given a single image the mean of
    % its blocks will be 0)
    trainingSet{idx} = imgY-mean(mean(imgY));
    idx = idx + 1;
    
end

% some images appear to be quite gray and low contrast.. that is because 
% when we visualize a matrix of double, matlab maps the lowest (negative) 
% value into 0 and the highest (positive) value into 255!

if nImagesTrain <= 10 || mod(nImagesTrain, 10) == 0
    
    % reshape the cell vector to form a huge matrix (image)
    trainingSetMat  = reshape(trainingSet, 10, 15);
    trainingSetImg  = cell2mat(trainingSetMat);
    
    trSetFig = figure(1);
    set(trSetFig, 'name', 'training set visualization', 'numbertitle', 'on');
    imshow(trainingSetImg, []);
else
    fprintf('Full visualization of the training set is not possible due to its structure.\n');
end

%% TRAINING SET BLOCK HANDLING

% then convert to a (column) vector to allow block division
trainingSetImg   = cell2mat(trainingSet);
[trRows, trCols] = size(trainingSetImg);

trBlocksMat    = mat2cell(trainingSetImg, bd * ones(1, trRows/bd), bd * ones(1, trCols/bd));

% unwrap the matrix of cells, obtaining a vector of 4x4 blocks
trBlocksVec = trBlocksMat(:);

% number of blocks, i.e. of vectors in the training set
N = length(trBlocksVec);

%% INITIALIZATION OF THE CODEBOOK

K = input('How many codewords in the codebook? (default is 64): ');

if isempty(K)
    % default number of codewords
    K = 64;
end

% useful when saving the codebook 
actualTime = clock;
hourStr = sprintf('%d-Jun-%d_%d', actualTime(3), actualTime(4), actualTime(5));
fileName = strcat(sprintf('learned_codebook_K%d_', K), hourStr);

% initialize the data structure that will contain the codebook
codebook = cell(K, 1);

alreadyChosen = zeros(1, K);

for k = 1:K

    idx = randi(K);
    
    while ismember(idx, alreadyChosen) 
        idx = randi(K);
    end
    
    temp = cell2mat(trBlocksVec(idx)); 
    
    codebook{k} = temp/norm(temp); 

    alreadyChosen(k) = idx;
    
end


%% K-SVD CODEBOOK CONSTRUCTION

% for each unwrapped block, we compute its sparse representation in terms
% on the columns of D. T0 defines how sparse is the representation 
% (that is the number of nonzero coefficients that are allowed).

T0 = input('How many codewords at max to be used for each y_i? (min 1, max K, default is 4 or 10): ');

if isempty(T0)
    if K >= 10
        T0 = 10;
    else
        T0 = 4;
    end
end

% transform the cell object into a matrix that we can pass as parameter to
% the orthogonal matching pursuit function
D = reshape(cell2mat(codebook), size(cell2mat(codebook(1)), 1)^2, size(codebook, 1));

% (just to be consistent with the paper notation), unwrap each block and
% build a (bd*bd)x(nBlocks) matrix (training set)

Y = zeros(bd*bd, N);

for i = 1:N
    block = cell2mat(trBlocksVec(i));
    Y(:, i) = block(:);
end

nIter = input('Input the number of K-SVD iterations (default is 10): ');

if isempty(nIter)
    nIter = 10;
end

% control variable on usage of a codeword
wasUnused = zeros(K, 1);

% limit of non-usage after which every codeword is re-initialized
reinitLim = 2;

for ksvdIter = 1:nIter
    
    fprintf('K-SVD iteration number %d out of %d\n', ksvdIter, nIter);
    % .. compute X keeping D fixed.

    sparseCodingProgress = waitbar(0, 'Sparse coding phase... 0%% done', 'name',...
        sprintf('Sparse coding progress, iter n. %d out of %d', ksvdIter, nIter));
    
    X = sparseCoding(Y, D, T0, 'omp', sparseCodingProgress);

    % if a codewords remains unused for more than a certain number of
    % iteration, then
    wasUnused = wasUnused + (sum(X, 2) == 0);
    
    % At the end of this phase, the k-th column of the X matrix contains all 
    % the weights to give to each column of D to obtain the best approximation, 
    % with a certain sparseness (only T0 non-zero coeff.s!) of the k-th 
    % training set sample (k-th column of Y).
    % So basically in X(1, 1) we find the "best" weight to give to the codeword
    % D(:, 1) to obtain Y(:, 1), in X(2, 1) the "best" weight to give to the 
    % codeword D(:, 2) to obtain Y(:, 1).. etc
    % In this contest, the quantity:
    %
    %               Y(:, k) ~= D*X(:, k)
    %
    % gives the "best" approximation in the sense that X(:, k) to the optimal
    % x: even if (since it's still an NP problem, 0-norm leads to a non-convex 
    % space...) we don't know how to compute the exact solution, it can be 
    % proved that MP, OMP, FOCUSS and other strategies lead to a very good
    % solution.

    % Once X is given, we modify the k-th column of the coodebook D (for each k
    % from 1 to K) to lower the MSE. To do this we exploit the singular value 
    % decomposition (SVD) applied on the reduced X and Y, that are the matrices
    % that contain respectively only the training set signals (Y) and the 
    % coefficients (or weights, in X) in which the k-th column of D/k-th 
    % codeword is used. So for each k, we compute the reduced matrices and 
    % apply the SVD:

    codebookUpdateProgress = waitbar(0, 'Codebook update phase... 0%% done', 'name',...
        sprintf('Fit phase progress, iter n. %d out of %d', ksvdIter, nIter));
    
    D = svdDictionaryUpdate(Y, D, X, codebookUpdateProgress);
    
    % save the actual codebook
    save(strcat(basePath, 'learned_codebooks/', fileName), 'D')
    
    % after the update and the codebook is saved, if any of the codewords
    % is not being used for more than reinitLim, re-initialize them by 
    % extracting a R.V. of iid gaussian r.v.s with mean 0 and var = 2
    for k = 1:K
       if wasUnused(k) > reinitLim
           % DEBUG
           fprintf('   - codeword %d was re-initialized due to non-usage\n', k)
           % variance = 2
           temp = 2*rand(bd*bd, 1);
           D(:, k) = temp/norm(temp);
           
           wasUnused(k) = 0;
       end
    end

end
