clc; clear all;

addpath(genpath('LMELTV'))
addpath(genpath('LTELTV'))
addpath quality_assess\

%% Load HSI data
isSimulation = false; %If preset_datasets is set to any of the values of case1, case2, case3, case4, or to any other custom simulation data set, then it is set to true; otherwise, it is set to false.
use_preset_datasets = true;
preset_datasets = 'Indian_pines_normal'; % Select from case1, case2, case3, case4, Urban_normal, Indian_pines_normal

model = 'LTELTV'; % Select from LMELTV, LTELTV

if use_preset_datasets == true
    load(strcat('data\',preset_datasets));
    [M,N,p] = size(Noisy_Img);
    switch preset_datasets
        case 'case1' 
            mat_rank = 10;
            ten_rank = [120,120,10];
            gamma_M = 0.03; 
            lambda_M = 60/sqrt(M*N);
            w_M = [1.05,0.95,0.25];
            gamma_T = 0.54;
            lambda_T = 1000/sqrt(M*N);
            w_T = [1.05,0.95,0.25];
        case 'case2'
            mat_rank = 10;
            ten_rank = [120,120,10];
            gamma_M = 0.05; 
            lambda_M = 75/sqrt(M*N);
            w_M = [1.05,0.95,0.25];
            gamma_T = 0.8;
            lambda_T = 1000/sqrt(M*N);
            w_T = [0.93,0.83,0.43];
        case 'case3'
            mat_rank = 10;
            ten_rank = [120,120,10];
            gamma_M = 0.035; 
            lambda_M = 65/sqrt(M*N);
            w_M = [1.05,0.95,0.25];
            gamma_T = 0.51;
            lambda_T = 1000/sqrt(M*N);
            w_T = [1.05,0.95,0.25];
        case 'case4'
            mat_rank = 10;
            ten_rank = [120,120,10];
            gamma_M = 0.55; 
            lambda_M = 700/sqrt(M*N);
            w_M = [0.93,0.83,0.43];
            gamma_T = 0.82;
            lambda_T = 1000/sqrt(M*N);
            w_T = [0.93,0.83,0.43];
        case 'Urban_normal'         
            mat_rank = 6;
            ten_rank = [276,276,9];
            gamma_M = 0.55; 
            lambda_M = 700/sqrt(M*N);
            w_M = [0.93,0.83,0.43];
            gamma_T = 0.7;
            lambda_T = 1000/sqrt(M*N);
            w_T = [0.93,0.83,0.43];
        case 'Indian_pines_normal' 
            mat_rank = 8;
            ten_rank = [105,105,8];
            gamma_M = 0.45; 
            lambda_M = 300/sqrt(M*N);
            w_M = [0.93,0.83,0.43];
            gamma_T = 0.82;
            lambda_T = 1000/sqrt(M*N);
            w_T = [0.93,0.83,0.43];
    end
else
    load('data\case4'); % You can change it to the desired dataset name by yourself.
    mat_rank = 10;
    ten_rank = [120,120,10];
    gamma_M = 0.55; 
    lambda_M = 700/sqrt(M*N);
    w_M = [0.93,0.83,0.43];
    gamma_T = 0.82;
    lambda_T = 1000/sqrt(M*N);
    w_T = [0.93,0.83,0.43];
end

tic
switch model
    case 'LMELTV' 
        disp('=============== LMELTV ===============')
        [output_image] = LMELTV(Noisy_Img,gamma_M, lambda_M, mat_rank, w_M);
    case 'LTELTV' 
        disp('=============== LTELTV ===============')
        [output_image] = LTELTV(Noisy_Img,gamma_T, lambda_T, ten_rank, w_T);
end
toc

output_image(output_image>1)=1;
output_image(output_image<0)=0;


%% quality assess
if isSimulation == true
    [q_psnr_mean,q_psnr] = MPSNR(Img,output_image);
    [q_ssim_mean,q_ssim] = MSSIM(Img,output_image);
    [q_fsim_mean,q_fsim] = MFSIM(Img,output_image);
    q_ergas = ErrRelGlobAdimSyn(255*Img,255*output_image);
    fprintf('psnr = %.4f , ssim = %.4f, fsim = %.4f , ergas = %.4f \n',q_psnr_mean,q_ssim_mean,q_fsim_mean,mean(q_ergas));
    if use_preset_datasets == true
        final_Img = cat(3, Img(:,:,50), Img(:,:,92), Img(:,:,164));
        final_output_image = cat(3, output_image(:,:,50), output_image(:,:,92), output_image(:,:,164));
        figure;
        subplot(1,2,1); imshow(final_Img); title('Clean Img');
        subplot(1,2,2); imshow(final_output_image); title('Output Img');
    end
else
    if use_preset_datasets == true
        switch preset_datasets
            case 'Indian_pines_normal'
                final_Noisy_Img = cat(3, Noisy_Img(:,:,149), Noisy_Img(:,:,30), Noisy_Img(:,:,1));
                final_output_image = cat(3, output_image(:,:,149), output_image(:,:,30), output_image(:,:,1));
            case 'Urban_normal'
                final_Noisy_Img = cat(3, Noisy_Img(:,:,63), Noisy_Img(:,:,102), Noisy_Img(:,:,208));
                final_output_image = cat(3, output_image(:,:,63), output_image(:,:,102), output_image(:,:,208));
        end 
        figure;
        subplot(1,2,1); imshow(final_Noisy_Img); title('Noisy Img');
        subplot(1,2,2); imshow(final_output_image); title('Output Img');
    end
end