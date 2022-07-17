function [ scrambledImage ] = phaseScrambleImage( inputImage )
%function [ scrambledImage ] = phaseScrambleImage( image )
%
%This function takes an image and keeps its fouier power spectrum constant
%but replaces the phase spectrum with uniform noise.
%
%Input:
%inputImage: input image
%
%Output:
%scrambledImage: output phase scrambled.
%

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

inFourier = fft2(inputImage);
inAmp = abs(inFourier);

%This uses a trick to easily scramble phases 
%Making the correct random fft matrix is a little tricky because 
%fourier transforms of real images have symmetry
%It's easier just to take the fourier transform of a white noise image
%White noise has a flat power spectrum and uniform phase spectrum
outPhase=angle(fft2(randn(size(inputImage))));

%reconstruct the scrambled image from its complex valued matrix
scrambledImage=ifft2(inAmp.*exp(1i.*outPhase),'symmetric');

end

