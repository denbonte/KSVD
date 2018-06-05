function [ksvdCoeffs, imgKSVD] = ksvd(img, D, bd, varargin)
%KSVD(img, D, bd, T0) returns the KSVD coefficients of the image passed as 
%parameter and optionally the reconstructed image by using T0 coeff.s per
%block.
%   Given a codebook D, compute the sparse representation of img in the D
%   space. If T0 is not specified, then the full KSVD is computed, meaning
%   that imgKSVD = img.
    
    K = size(D, 2);
    
    if nargin == 4
        T0 = varargin{1};
    else
        T0 = K;
    end
    
    img = double(img);
    
%     if nargin == 1
%         disp('yay');
%     end

    %% BLOCK DIVISION
    
    [rows, cols, chs] = size(img);
    
    if chs > 1
       error('dimension mismatch: img must be an NxM matrix (1 channel)');
    end
    
    % as usually, divide the image in (bd)x(bd) blocks using cells logic: 
    % fairly simple, using mat2cell - it let us specify the size of each vector
    % in the dynamic structure, i.e. array of cells
    % If we issued the instruction as "mat2cell(img, bd, bd)" the result would
    % have provoked an error: the function tries to store the image into an 4x4
    % vector inside a cell.. while "bd * ones(1, rows/bd)" basically makes the
    % instruction divide the image row-wise into blocks of length bd.
    % How many blocks do we need to "span" the image horizontally? rows/bd!
    % (assuming of course bd is a power of 2 or simply k * bd)
    % NOTE: structure is preserved
    blocksMat = mat2cell(img, bd * ones(1, rows/bd), bd * ones(1, cols/bd));

    % unwrap the matrix of cells, obtaining a vector of 4x4 blocks
    blocksVec = blocksMat(:);

    % number of blocks, i.e. of vectors in the training set
    imgN = length(blocksVec);

    imgY = zeros(bd*bd, imgN);

    for i = 1:imgN
        block = cell2mat(blocksVec(i));
        imgY(:, i) = block(:);
    end


    %% COEFFICIENTS ESTIMATION (main)

    % pre-allocate memory for the coefficients matrix
    ksvdCoeffs = zeros(K, imgN);

    coeffEstimationProgress = waitbar(0, 'KSVD Coefficient estimation: 0%% done', 'name',...
        sprintf('KSVD, T0 = %d', T0));

    for i = 1:imgN

        waitbar(i/imgN, coeffEstimationProgress, sprintf('KSVD Coefficient estimation: %2d%% done',...
            round(100*i/imgN)));    

        % sparse coding on the i-th training set vector
        ksvdCoeffs(:, i) = omp(D, imgY(:, i), T0);

    end

    close(coeffEstimationProgress)

    %% IMAGE RECONSTRUCTION (optional)

    Yksvd = D*ksvdCoeffs;

    blocksVecKSVD = cell(imgN, 1); 

    for i = 1:imgN

        % reshape every (bd*bd)x1 vector into a (bd)x(bd) matrix, store it as a
        % cell in a cell matrix
        blocksVecKSVD{i} = reshape(Yksvd(:, i), bd, bd);
    end

    % being a row vector or a column vector, it doesn't matter: this
    % instruction will shape the cells array as the original image again
    recBlocksMat = reshape(blocksVecKSVD, rows/bd, cols/bd);

    % we can then convert the matrix of cells directly into a big 
    % (rows)x(cols) matrix => cells are awesome.
    imgKSVD = cell2mat(recBlocksMat);


end

