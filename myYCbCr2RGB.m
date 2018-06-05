function imgRGB = myYCbCr2RGB(imgYCbCr)
%myYCbCr2RGB (MICD) - converts a yCbCr image to a RGB one.
%   Returns a three-channel RGB image.

    [rows, cols, chs] = size(imgYCbCr);
    
    R = zeros(rows, cols);
    G = zeros(rows, cols);  
    B = zeros(rows, cols);
    
    if chs ~= 3
        error('ERROR: the input image must have 3 channels (YCbCr).');
        return;
    end
    
    %% Conversion
    
    Y  = double( imgYCbCr(:, :, 1) );
    Cb = double( imgYCbCr(:, :, 2) );
    Cr = double( imgYCbCr(:, :, 3) );
    
    R = uint8( Y  + 1.4025*(Cr - 128) );
    G = uint8(1.000*Y - 0.344*(Cb - 128) - 0.7142*(Cr - 128) );
    B = uint8(1.000*Y + 1.773*(Cb - 128) );
    
    imgRGB = cat(chs, R, G, B);

end

