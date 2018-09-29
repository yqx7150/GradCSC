function [Iout1,param1] = GradCSC(I1,Q1,sigma,sigmai,GradCSC)
%                 - InputPSNR : PSNR of fully sampled noisy reconstruction
%                 - PSNR0 : PSNR of normalized zero-filled reconstruction
%                 - PSNR : PSNR of the reconstruction at each iteration of the DLMRI algorithm
%                 - HFEN : HFEN of the reconstruction at each iteration of the DLMRI algorithm
%                 - itererror : norm of the difference between the reconstructions at successive iterations
%                 - Dictionary : Final DLMRI dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu2 = GradCSC.mu2;
num = GradCSC.num;
lambda_residual = GradCSC.lambda_residual;
lambda = GradCSC.lambda; %
verbose_admm = GradCSC.verbose_admm;
max_it = GradCSC.max_it;
kernels = GradCSC.kernels;
sigma2=sqrt((sigmai^2)+(sigma^2));  %Effective k-space noise level 
La2=140/(sigma2); % \nu weighting of paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MAIN CODE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I1=double(I1(:,:,1));
I1=I1/(max(max(I1))); %Normalize input image
[aa,bb]=size(I1);     %Compute size of image

DZ=((sigma2/sqrt(2))*(randn(aa,bb)+(0+1i)*randn(aa,bb)));  %simulating noise
I5=fft2(I1);          %FFT of input image
I5=I5+DZ;             %add measurement noise in k-space (simulated noise)

%Compute Input PSNR after adding noise
IG=abs(ifft2(I5));
InputPSNR=20*log10((sqrt(aa*bb))*(max(max(abs(IG))))/norm(double(abs(IG))-double(I1),'fro'));
param1.InputPSNR=InputPSNR;
index=find(Q1==1); %Index the sampled locations in sampling mask
I2=(double(I5)).*(Q1);  %Apply mask in DFT domain
I11=ifft2(I2);          % Inverse FFT - gives zero-filled result
%initializing simulation metrics
I11=I11/(max(max(abs(I11))));  %image normalization
% finite diff
sizeF = size(I11);
C.eigsDtD = abs(psf2otf([1,-1],sizeF)).^2 + abs(psf2otf([1;-1],sizeF)).^2;
TVD = @(U) ForwardD(U);
TVDt = @(X,Y) Dive(X,Y);
bregc1 = zeros(sizeF);
bregc2 = bregc1;
%GradCSC iterations
for kp=1:num    
    I11=abs(I11); % I11=I11/(max(max(I11)));
    Iiter=I11;    
    [D1X,D2X] = TVD(I11);    % finite diff

    %% subproblem-1-2 %%%%%%%%%%%%%%%%%%
    [z, I3n1] = admm_solve_conv2D_weighted_GradCSC(D1X, kernels.d, ones(size(D1X)), lambda_residual, lambda, max_it, 1e-3, verbose_admm); 
    [z, I3n2] = admm_solve_conv2D_weighted_GradCSC(D2X, kernels.d, ones(size(D2X)), lambda_residual, lambda, max_it, 1e-3, verbose_admm); 
    I3n1 = (I3n1+mu2*(D1X + bregc1))/(1+mu2);  %gradient-image variable result
    I3n2 = (I3n2+mu2*(D2X + bregc2))/(1+mu2);  %gradient-image variable result
    
    %% subproblem-3 %%%%%%%%%%%%%%%%%%    
    I3n = ifft2(fft2(TVDt(I3n1-bregc1,I3n2-bregc2))./(C.eigsDtD+eps));
    inn = abs(I3n)>1;I3n(inn)=1; 
    I2=fft2(I3n);  %Move from image domain to k-space    
    %K-space update formula
    if(sigma2<1e-10)
        I2(index)=I5(index);
    else
        I2(index)= (1./(mu2*C.eigsDtD(index)+La2)).*(mu2*C.eigsDtD(index).*I2(index) + La2*I5(index));
    end 
    I11=ifft2(I2);   %Use Inverse FFT to get back to image domain
    inn2= abs(I11)>1;I11(inn2)=1;
    % ==================%  Update % ==================
    bregc1 = bregc1 + (D1X - I3n1);
    bregc2 = bregc2 + (D2X - I3n2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %Compute various performance metrics
    HFEN(kp)=norm(imfilter(abs(I11),fspecial('log',15,1.5)) - imfilter(I1,fspecial('log',15,1.5)),'fro');
    PSNR(kp)=20*log10(sqrt(aa*bb)*255/norm(double(abs(I11))-double(I1),'fro')) -(20*log10(255));
    fprintf(1, 'Iter=%d, PSNR=%f, HFEN=%f\n', kp, PSNR(kp),HFEN(kp)); 
    figure(11);imshow(abs(I11),[] );iptsetpref('ImshowBorder','tight');
end
Iout1=abs(I11);
param1.PSNR=PSNR;
param1.HFEN=HFEN;
% param1.itererror=itererror;
