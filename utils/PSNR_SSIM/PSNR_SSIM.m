
dataset = 'Set5';
upscale = 4;
HRfolder = sprintf('../benchmark/%s/HR', dataset);
SRfolder = sprintf('../../BDIR/BiSRx4-results/BiSRx4-sup/%s_SSNet_RGB_BiSR_x4_read0.0_shot0', dataset);%sprintf('../../LSGNet/ablation/GAN-l2-img+diff/%s', dataset);
Logfolder = fullfile(SRfolder, 'psnr_ssim.txt');
if exist(Logfolder, 'file')
    delete(Logfolder);
end
HR = dir(char(strcat(HRfolder, '/', '*.png')));
SR = dir(char(strcat(SRfolder, '/', '*.png')));
HR_names = {HR.name};
HR_names = HR_names(1:length(HR_names));
SR_names = {SR.name};
SR_names = SR_names(1:length(SR_names));
mean_psnr = [];
mean_ssim = [];
file = fopen(Logfolder, 'wt');
for n=1:length(HR_names)
    HR_image = imread(char(strcat(HRfolder, '/', HR_names(n))));
    SR_image = imread(char(strcat(SRfolder, '/', SR_names(n))));
    [psnr, ssim] = compute_diff(HR_image, SR_image, upscale);
    HR_image = modcrop(HR_image, upscale);
    abs_diff = abs(double(SR_image) - double(HR_image));
    norm_abs_diff = abs_diff / max(max(max(abs_diff)));
    imwrite(uint8(norm_abs_diff*255), char(HR_names(n)));
    s = sprintf('%s : %0.2f  %0.4f \r\n', string(HR_names(n)), psnr, ssim);
    fprintf(file, '%s', s);
    mean_psnr(end + 1) = psnr;
    mean_ssim(end + 1) = ssim;
    sprintf(char(HR_names(n)))
end
mean_psnr = sum(mean_psnr) / length(mean_psnr);
mean_ssim = sum(mean_ssim) / length(mean_ssim);
s = sprintf('Mean: %0.2f  %0.4f  ', mean_psnr, mean_ssim)
fprintf(file, '%s', s);
fclose(file);
