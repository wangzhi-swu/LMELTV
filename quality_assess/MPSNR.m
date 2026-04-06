function [mpsnr, psnr] = MPSNR(OriHSI, ResHSI)
% this function is to calculate the MPSNR of the restoration performance
%OriHSI is the true hyperspectral image with L*(nr*nc) dims
%ResHSI is the restorated image with the same dims

[M,N,L] = size(OriHSI);
[M1,N1,L1] = size(ResHSI);
if L~=L1 || N~=N1 ||M~=M1
    disp(' The dims of the two matrix must be same!');
%     exit;
end

for i= 1:L
    diff_img = 255*(OriHSI(:,:,i)-ResHSI(:,:,i));
    psnr(i) = 10*log10(255^2/mse(diff_img));
end
mpsnr = sum(psnr) / L;