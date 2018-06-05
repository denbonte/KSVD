function [Y, Cb, Cr] = myRGB2YCbCr(img)
%myRGB2YCbCr (MICD) - converts a RGB image to a YCbCr one.
%   Returns the three channels Y, Cb, Cr separately.

    [rows, cols, chs] = size(img);
    
    Y  = zeros(rows, cols);
    Cb = zeros(rows, cols);
    Cr = zeros(rows, cols);
    
    if chs ~= 3
        error('ERROR: the input image must have 3 color channels (RGB).');
        return;
    end

    %% CONVERSION

    R = double( img(:, :, 1) );
    G = double( img(:, :, 2) );
    B = double( img(:, :, 3) );
    
    Y  = uint8( 0.299*R + 0.587*G + 0.144*B );
    Cb = uint8( -0.169*R -0.331*G + 0.500*B + 128);
    Cr = uint8( 0.500*R - 0.419*G - 0.081*B + 128);

end

