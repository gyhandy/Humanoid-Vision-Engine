% function imScrambled = imscramble(im,p,rescale)
% 
% This function creates Fourier-scrambled images. These images can be
% scrambled to different degrees. There is an option to rescale the image
% to the original range, which can slightly change luminance or contrast.
% One alternative possibility to deal with this is to rescale images
% manually using e.g. the luminance and root-mean-square contrast of the
% original image.
%
% Input:
%   im: input image (can be black&white [i.e. 2D] or with RGB color [i.e.
%    3D]), ideally with range 0 to 1.
%   p (optional): scrambling factor (between 0 and 1) [default: 1]
%   rescale (optional): 'off': no rescaling,
%                     'range': rescaling to original range,
%                     'cutoff': all values exceeding the min/max of the
%                       original image are set to these values
%                       [default 'off']
%
% Warning: Rescaling may change the luminance or contrast! This does not
% occur with 'off', but with that option out-of-range values are possible
% through scrambling. This can be prevented by choosing images without
% maximal contrast (e.g. ranging from 0.1 to 0.9).
%
% Example:
% load mandrill
% X = ind2rgb(X,map);
% h = figure; p = get(h,'Position'); set(h,'Position',[0.2 1 2 1].*p); xlabel('Original')
% subplot(1,3,1), image(X); axis equal
% Xscrambled = imscramble(X,0.6,'cutoff');
% Xscrambled2 = imscramble(X,0.6,'range');
% subplot(1,3,2), image(Xscrambled);  axis equal; xlabel('Scrambled with ''cutoff''')
% subplot(1,3,3), image(Xscrambled2);  axis equal; xlabel('Scrambled with ''range''')
%
% by Martin Hebart (2009)

function imScrambled = imscramble(im,p,rescale)

if nargin < 3
    rescale = 'off';
end

if nargin < 2
    p = 1;
end

imclass = class(im); % get class of image

im = double(im);
imSize = size(im);

RandomPhase = p*angle(fft2(rand(imSize(1), imSize(2)))); %generate random phase structure in range p (between 0 and 1)
RandomPhase(1) = 0; % leave out the DC value

if length(imSize) == 2
    imSize(3) = 1;
end

% preallocate
imFourier = zeros(imSize);
Amp = zeros(imSize);
Phase = zeros(imSize);
imScrambled = zeros(imSize);

for layer = 1:imSize(3)
    imFourier(:,:,layer) = fft2(im(:,:,layer));         % Fast-Fourier transform
    Amp(:,:,layer) = abs(imFourier(:,:,layer));         % amplitude spectrum
    Phase(:,:,layer) = angle(imFourier(:,:,layer));     % phase spectrum
    Phase(:,:,layer) = Phase(:,:,layer) + RandomPhase;  % add random phase to original phase
    % combine Amp and Phase then perform inverse Fourier
    imScrambled(:,:,layer) = ifft2(Amp(:,:,layer).*exp(sqrt(-1)*(Phase(:,:,layer))));
end

imScrambled = real(imScrambled); % get rid of imaginery part in image (due to rounding error)

switch lower(rescale)
    case 'range'
        minim = min(im(:)); maxim = max(im(:));
        imScrambled = minim+(maxim-minim).*(imScrambled-min(imScrambled(:)))./(max(imScrambled(:))-min(imScrambled(:))); % rescale to original
    case 'cutoff'
        minim = min(im(:)); maxim = max(im(:));
        imScrambled = min(imScrambled,maxim); % adjusts range for maximum
        imScrambled = max(imScrambled,minim); % adjusts range for minimum
end

imScrambled = cast(imScrambled,imclass); % bring image back to original class