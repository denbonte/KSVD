function [updD] = svdDictionaryUpdate(Y, D, X, codebookUpdateProgress)
%SVDDICTIONARYUPDATE updates the dictionary (i.e. codebook) using an SVD based
%technique.
%   SVDDICTIONARYUPDATE(D, X, Y) updates each column of D (i.e. codewords)
%   exploiting the SVD decomposition technique. The method implemented is
%   the one described in Elad et al. (2006):
%       - first, the k-th column is removed from the codebook and a
%       reconstruction of Y is computed. The error matrix of this very 
%       approximation is computed (element by element difference);
%       - the obtained matrix is reduced and then decomposed using the 
%       Singular Value Decomposition;
%       - the updated d and x are computed from the left-singular vectors
%       matrix, the right-singular vector matrix and the singular values.
%
%   See Elad et al. for further details.
%
%   codebookUpdateProgress handles graphics.

    K = size(D, 2);
    N = size(Y, 2);

    for k = 1:K

        waitbar(k/K, codebookUpdateProgress, sprintf('Codebook update phase... %2d%% done',...
            round(100*k/K)));   
        
        % first of all we compute the error matrix Ek, that is the 
        % approximation of Y by using all the codewords of D but the k-th and
        % the coefficients we previously computed in the sparse coding phase
        Dk = D;
        Dk(:, k) = [];

        Xk = X;
        Xk(k, :) = [];

        Ek = Y - Dk*Xk;

        % then, we check the positions of the T0 (at maximum) non-zero 
        % coefficients of X(:, k), save their positions in a vector w
        w = find(X(k, :));

        reductionMatrix = zeros(N, length(w));
        % then build the reduction matrix by setting to one only the positions
        % (w(i), i)
        for i = 1:length(w)
            reductionMatrix(w(i), i) = 1;
        end

        % obtain the Yreduced matrix and the Xreduced vector by multiplying
        % them by the reduction matrix itself (superfluous)
% %         Yr = Y*reductionMatrix;
% %         xr = X(k, :)*reductionMatrix;

        % obtain the reduced Ek matrix
        Ekr = Ek*reductionMatrix;

        if isempty(Ekr)
            continue;
        end
        
        % apply the SVD to the Ekr matrix
        [U, delta, Vtransp] = svd(Ekr);
        V = Vtransp';

        % obtain the optimal d as the first column of U (first left-singular
        % vector)
        dNew = U(:, 1);

        % obtain the new reduced coefficient vector as the first column of V
        % (first right-singular vector) multiplied by delta(1, 1) (first
        % singular value)
        xrNew = V(:, 1)*delta(1, 1);

        % and then we must expand back (to X(k, :) dimension) the newly 
        % obtained reduced coefficient vector
        xNew = zeros(size(X(k, :)));

        idx = 1;
        for i = w
           xNew(i) = xrNew(idx);
           idx = idx+1;
        end

        % update the codebook and the coefficient matrix
        D(:, k) = dNew;
        X(k, :) = xNew;

    end
    
    updD = D;
    
    close(codebookUpdateProgress);

end

