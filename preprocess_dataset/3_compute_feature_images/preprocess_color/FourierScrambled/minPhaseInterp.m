function [sequence] = minPhaseInterp(startImage,endImage,interpVals)
%function [sequence] = minPhaseInterp(startImage,endImage,interpVals)
%
%This function creates a morph between startImage and endImage.
%It does this by interpolating the phase of the fourier transform between
%the two images. It is careful to treat phase correctly as a circular
%variable and it interpolates in the minimum phase direction with equal
%angle steps(Ales, Farzin et al. Journal Of Vision 2012).
%
%WARNING: All output images have the power spectrum of the final image.
%
%startImage: 2d image to start the sequence. This image's power spectrum is
%repaced by endImage power spectrum
%
%endImage: 2d image to end the sequence on.  Entire sequences uses this
%images power spectrum
%
%interpVals: A vector containing the phase steps for the sequence. 0 is startImage phase, 1 is
%endImage phase.  .5 is halfway between both phases.
%
%For more details see:
%An objective method for measuring face detection thresholds using the 
%sweep steady-state visual evoked response.
%Ales, JM*, Farzin F*, Rossion B, Norcia AM
%(2012) Journal of Vision 12(10):18, 1?18
%
%
%Example:
%
%imFinal=imread('eight.tif');
%imStart=randn(size(imFinal));
%imSeq=minPhaseInterp(imStart,imFinal,linspace(0,1,10));
%figure;
%colormap gray;
%for iSeq=1:10,
%  imagesc(imSeq(:,:,iSeq));
%  pause(.1);
%end

%email: justin.ales@gmail.com
%Copyright 2012 Justin Ales
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.%

startImage = double(startImage);
endImage = double(endImage);

sizeStart = size(startImage);
sizeEnd =   size(endImage);

if length(sizeStart)>2 || length(sizeEnd)>2
    error('Only works for 2d gray scale images.')
end


if sizeStart(1) ~= sizeEnd(1) || sizeStart(2) ~= sizeEnd(2)
    error('Images are not equal sizes. Make start and ending images equal dimensions')
end

if max(interpVals)>1 || min(interpVals) <0
    error('interpolation values must be between 0 and 1');
end

%setup interpolation function
x = 1:sizeStart(1);
y = 1:sizeStart(2);
z = [0 1];


%Take fourier transform of input images and decompopse complex values
%to phase and amplitude
%Use amplitude spectrum for the end image on the first image.
%This keeps the amplitude spectrum constant.
startFourier = fft2(startImage);
endFourier   = fft2(endImage);

startPhase   = angle(startFourier);
%startAmp     = abs(startFourier);

endPhase     = angle(endFourier);
endAmp       = abs(endFourier);

startAmp = endAmp;

initialSequence = cat(3,startPhase,endPhase);

%This is where I figure out the minimum phase direction.
%I do this by chaining some trigonometry operations together.
%D is the angle between the starting and ending phase
%We know we want to change phase by this amount
%We then redefine the starting phase so the interpolation always goes in
%the correct direction
D = squeeze(initialSequence(:,:,1)-initialSequence(:,:,2));
delta = atan2(sin(D),cos(D));
initialSequence(:,:,1) = initialSequence(:,:,2) + delta;

%This is slow, but it's easy and I'm lazy.
[xi yi zi] = ndgrid(x,y,interpVals);
phaseSequence = interpn(x,y,z,initialSequence,xi,yi,zi);

% phaseSequence(phaseSequence>pi) = phaseSequence(phaseSequence>pi) -2*pi;
% phaseSequence(phaseSequence<pi) = phaseSequence(phaseSequence<pi) + 2*pi;

phaseSequence = mod(phaseSequence+pi,2*pi)-pi;

ampSequence = repmat(endAmp,[1 1 length(interpVals)]);

complexSequence = ampSequence.*exp(1i.*phaseSequence);

%'symmetric' flag is important here.
sequence = ifft2(complexSequence,'symmetric');