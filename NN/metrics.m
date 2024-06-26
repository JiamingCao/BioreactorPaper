load('test_processed.mat')
load('../SphericalInclusion/images3.mat')

recon1 = noisy_img(:,:,:,2401:end);
truth = clean_img(:,:,:,2401:end);
clear noisy_img clean_img

iou1 = zeros(size(truth,4), 1);
iou2 = zeros(size(truth,4), 1);
thresh = 1;

for i=1:length(iou1)
    tmp1 = squeeze(recon1(:,:,:,i));
    tmp2 = squeeze(recon2(:,:,:,i));
    tmp0 = squeeze(truth(:,:,:,i));

    iou1(i) = sum((tmp1(:)>thresh) .* (tmp0(:)>0) .* mask(:)) / sum((((tmp1(:)>thresh) + (tmp0(:)>0))>0) .* mask(:));
    iou2(i) = sum((tmp2(:)>thresh) .* (tmp0(:)>0) .* mask(:)) / sum((((tmp2(:)>thresh) + (tmp0(:)>0))>0) .* mask(:));
end
