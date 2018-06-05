function [ imgDCT ] = myDCT(img, bd, coeffs)
%MYDCT(img, bd, coeffs) implements the DCT transform and reconstruct the
%image using a limited number of coefficients.
%   myDCT exploits the built-in blockproc and dctmtx to compute the DCT 
%   transform of img, that is then reconstructed masking the result of the 
%   transform with a zig-zag matrix that has only the first coeffs 
%   coefficients different from zero.
%   "bd" is the block dimension (JPG uses 8x8 blocks).
%   Taken from "https://it.mathworks.com/help/images/discrete-cosine-transform.html"
%   and "https://blogs.mathworks.com/steve/2013/04/12/revisiting-dctdemo-part-4/"

    % 8x8 blocks, as JPEG does => bd = 8
    f = @(block) dct2(block.data);
    A = blockproc(img, [bd bd], f);
    
    % Compute DCT coefficient variances and decide
    % which to keep.
    B = im2col(A, [bd bd], 'distinct')';
    vars = var(B);
    [~, idx] = sort(vars, 'descend');
    keep = idx(1:coeffs);
    
    % Zero out the DCT coefficients we are not keeping.
    B2 = zeros(size(B));
    B2(:, keep) = B(:, keep);

    % Reconstruct image using 8-by-8 block inverse
    % DCTs.
    C = col2im(B2', [bd bd], size(img), 'distinct');
    finv = @(block) idct2(block.data);
    imgDCT = blockproc(C, [bd bd], finv);

end

