
%% The Code is created based on the method described in the following papers: 
% [1] J. Xiong , H. Lu , M. Zhang , Q. Liu, Convolutional Sparse Coding in Gradient Domain for MRI Reconstruction, ACTA AUTOMATICA SINICA, 43(10):1841-1849, 2017.
% Author: J. Xiong , H. Lu , M. Zhang , Q. Liu 
% Date : 4//2018 
% Version : 1.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2018, Department of Electronic Information Engineering, Nanchang University. 
% GradCSC - Convolutional sparse coding in Gradient domain
% 
% Paras: 
%       1. M0 : Input MR Image (real-valued)
%       2. Q1 : Sampling Mask (for 2D DFT data) with zeros at non-sampled locations and ones at sampled locations
%       3. sigma : Simulated noise level (standard deviation of simulated noise - added during simulation)
%       4. sigmai : Noise level (standard deviation of complex noise) in the DFT space of the peak-normalized input image.
%                   To be set to 0 if input image data is noiseless.
%       5. num : Number of iterations
%       6. Iout1 - Image reconstructed with GradCSC algorithm from undersampled data.
%       7. param1 - Structure containing various parameter values/performance metrics from simulation for GradCSC.

% Example 
% ========== 
%%%%%%%%%%%%%%%%%%%%%%%% step1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; 
close all;
path(path,'./utilizes/');

%%%%%%%%%%%%%%%%%%%%%%%% step2 %%%%%%%%%%%%
kernels = load('Gradientfilters_mri.mat');
d = kernels.d;
sqr_k = ceil(sqrt(size(d,3))); pd = 1;
psf_radius = floor(size(d,1)/2);
d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) =  d(:,:,j + 1);
end
imagesc(d_disp), colormap gray, axis image, colorbar, title('Kernels used');

%%%%%%%%%%%%%%%%%%%%%%%% step3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Q1.mat
Q1 = fftshift(Q1);
figure(1); imshow(Q1,[]);
n = size(Q1,2);
k = sum(sum(Q1));
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, k,1-k/n/n);

%#######%%%%% test image %%%%
M0 = imread('example.jpg');
M0 = im2double(M0);
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
if (max(M0(:))<2);   M0 = M0*255;    end
[min(M0(:)),max(M0(:))];

%%%%%%%%%%%%% GradCSC Reconstruction %%%%%%%%
I1 = M0;
Q1 = fftshift(Q1);
sigma = 10.2;sigmai = 10.2;   
GradCSC.mu2 = 2;
GradCSC.num = 50;
GradCSC.lambda_residual = 5.0;
GradCSC.lambda =0.1; 
GradCSC.verbose_admm = 'all';
GradCSC.max_it = 10;
GradCSC.kernels = kernels;
[Iout1,param1]=Grad_CSC(I1,Q1,sigma,sigmai,GradCSC);
A=imrotate(abs(Iout1),180,'nearest');
a=A(end:-1:1,end:-1:1);
I1=I1./max( abs(I1(:)));
Rec_im_1=a;
figure(42);
imshow(Rec_im_1);
figure(22);

    
