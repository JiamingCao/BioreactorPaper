load('test_processed.mat')
% load('../SimData/3D/images3.mat')
load('../SimulateData/images3.mat')
load('bnd.mat')

recon1 = noisy_img(:,:,:,2401:end);
truth = clean_img(:,:,:,2401:end);
clear noisy_img clean_img

cmap = hot(128); % colormap

thresh = 1; % Since our data is normalized, this effectively says we only care about voxels above one standard deviation

%%%% Change this to something you need
tmp=recon1(:,:,:,51);
%%%%

tmp(tmp<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol1 = volshow(tmp+bnd, Colormap=cmap);
cfg1 = vol1.Parent;
cfg1.CameraPosition=[0,-120,80];
cfg1.CameraUpVector=[0,0,1];
cfg1.BackgroundColor = [1,1,1];
cfg1.BackgroundGradient = 0;


