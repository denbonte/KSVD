function [X] = sparseCoding(Y, D, T0, mpAlgorithm, sparseCodingProgress)
%SPARSECODING implements sparse coding exploting matching persuit
%algorithms.
%   SPARSECODING(Y, D, T0) returns the sparse representation of Y with 
%   respect to the columns of D.
%
%   Y is the training set, i.e. the has in every column the signals we want
%   to predict using less codewords possible.
%   D is the codebook matrix containing a codeword in every column.
%   T0 is the masimum number of codewords we're allowed to use to predict
%   every column of Y.
%   
%   mpAlgorithm allows to choose the matching persuit algorithm to be used.
%   sparseCodingProgress handles the graphics.

    K = size(D, 2);
    N = size(Y, 2);

    % pre-allocate memory for the coefficients matrix
    X = zeros(K, N);

%     sparseCodingProgress = waitbar(0, 'Sparse coding phase... 0%% done', 'name',...
%         sprintf('Sparse coding progress, iter n. %d out of %d', ksvdIter, nIter));

    for i = 1:N

        waitbar(i/N, sparseCodingProgress, sprintf('Sparse coding phase... %2d%% done',...
            round(100*i/N)));    

        if strcmp(mpAlgorithm, 'omp') == 1
            % sparse coding on the i-th training set vector
            X(:, i) = omp(D, Y(:, i), T0);
        else
            if strcmp(mpAlgorithm, 'mp') == 1
                disp('matching persuit algorithm not implemented yet')
            end
                
        end

    end

    close(sparseCodingProgress)


end

