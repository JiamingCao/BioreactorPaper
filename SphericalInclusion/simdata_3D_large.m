addpath(genpath('/home/jcao/Documents/NIRFASTer'))
clear

mesh = load_mesh('cylinder_large');

%%
samples = 1;
mesh.muaf = zeros(size(mesh.muaf)); % no background fluorescence
num_nodes = size(mesh.nodes, 1);
max_blobs = 2;
blob_r_rng = [7,15];    % mm
% blob_muaf_rng = [3,7];   % times baseline
blob_muaf_rng = [1e-3,1e-1];   % mm-1; eta=0.4 in this mesh
boundary = [min(mesh.nodes(:,1)), max(mesh.nodes(:,1)), min(mesh.nodes(:,2)), max(mesh.nodes(:,2)), min(mesh.nodes(:,3)), max(mesh.nodes(:,3))];
radius = boundary(2);

all_muaf = zeros(size(mesh.nodes,1), samples);
all_datax = zeros(size(mesh.link,1), samples);
all_datafl = zeros(size(mesh.link,1), samples);
all_datax_clean = zeros(size(mesh.link,1), samples);
all_datafl_clean = zeros(size(mesh.link,1), samples);

all_x = zeros(max_blobs, samples);
all_y = zeros(max_blobs, samples);
all_z = zeros(max_blobs, samples);
all_nblob = zeros(samples,1);
all_noise = zeros(2, samples);
all_fluctuate = zeros(2, samples);

solver=get_solver('BiCGStab_GPU');
opt = solver_options;
opt.GPU = -1;

for rep = 1:samples
    fprintf('%d/%d\n', rep, samples);
    mesh2 = mesh;
    num_blob = randperm(max_blobs, 1);
    blob_r = rand(num_blob,1) * (blob_r_rng(2) - blob_r_rng(1)) + blob_r_rng(1);
    blob_muaf = rand(num_blob,1) * (blob_muaf_rng(2) - blob_muaf_rng(1)) + blob_muaf_rng(1);
    blob_x = nan;
    blob_y = nan;
    blob_z = nan;
    while 1
        idx = true(num_blob, num_blob);
        tmp = pdist2([blob_x, blob_y, blob_z], [blob_x, blob_y, blob_z]) > blob_r+blob_r'+2;
        if all(blob_x.^2+blob_y.^2 < (blob_r-radius).^2) && (num_blob ==1 || all(tmp(tril(idx,-1))))
            break
        end
        blob_x = rand(num_blob,1) * (boundary(2)-boundary(1)) + boundary(1);
        blob_y = rand(num_blob,1) * (boundary(4)-boundary(3)) + boundary(3);
        blob_z = rand(num_blob,1) .* (boundary(6)-boundary(5)-blob_r-2) + (boundary(5)+blob_r);
    end
    
    fluctuate = 0.1*(rand(2,1)-0.5); % fluctuate the background mua by +/-10%
    mesh2.muax = mesh.muax*(1+fluctuate(1));
    mesh2.muam = mesh2.muax;
    mesh2.musx = mesh.musx*(1+fluctuate(2));
    mesh2.musm = mesh2.musx;

    for i=1:num_blob
        blob=[];
        blob.x = blob_x(i);
        blob.y = blob_y(i);
        blob.z = blob_z(i);
        blob.r = blob_r(i);
        blob.muaf = blob_muaf(i);
        mesh2 = add_blob(mesh2, blob);
    end
    mesh2.kappax = 1./(3*(mesh2.muax + mesh2.musx));
    mesh2.kappam = 1./(3*(mesh2.muam + mesh2.musm));

    try
        data = femdata_fl(mesh2, 0, solver,opt);
    catch
        data = femdata_fl(mesh2, 0, solver,opt);
    end
    all_muaf(:,rep) = mesh2.muaf;
    noise_lv = rand(2,1)*0.01;
    all_noise(:,rep) = noise_lv;
    all_datafl_clean(:,rep) = data.amplitudefl;
    all_datax_clean(:,rep) = data.amplitudex;
    all_datafl(:,rep) = data.amplitudefl + noise_lv(1)*max(data.amplitudefl).*randn(size(data.amplitudefl));
    all_datax(:,rep) = data.amplitudex + noise_lv(2)*max(data.amplitudex).*randn(size(data.amplitudex));
    all_nblob(rep) = num_blob;
    all_x(1:num_blob,rep) = blob_x;
    all_y(1:num_blob,rep) = blob_y;
    all_z(1:num_blob,rep) = blob_z;
    all_fluctuate(:,rep) = fluctuate;
end

%% Prepare for NN
xgrid = linspace(-65,65,48);
ygrid = linspace(-65,65,48);
zgrid = linspace(-75,75,56);
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);

all_muaf2 = mesh.vol.mesh2grid*all_muaf;
clean_img = reshape(all_muaf2,48,48,56,samples);

[J, data0]= jacobiangrid_fl(mesh,[],[],[],0,solver, opt);
J = rmfield(J, 'completem');
[~, invop] = tikhonov(J.complexm./data0.amplitudex,100);
tmp = all_datafl_clean./all_datax_clean;
maxamp = max(tmp);
norm_noise = 0.02*rand(samples, 1);
noisestd = maxamp' .* norm_noise;
noise = abs(randn(size(tmp,1), samples)*diag(noisestd));
% noise(noise<0) = 1e-20;

recon = invop*(tmp + noise);
noisy_img = reshape(recon,48,48,56,samples);

inmesh = mesh.vol.gridinmesh;

for i=1:samples
    tmp=clean_img(:,:,:,i);
    clean_img(:,:,:,i)=tmp/std(tmp(:));
end
for i=1:samples
    tmp=noisy_img(:,:,:,i);
    noisy_img(:,:,:,i)=tmp/std(tmp(:));
end
mask=zeros(48,48,56);
mask(inmesh)=1;

save('images3', 'clean_img', 'noisy_img', 'inmesh','all_x', 'all_y', 'all_z', 'all_nblob', 'all_muaf', 'all_datafl', 'all_datax', 'all_noise', 'all_fluctuate', 'all_datax_clean', 'all_datafl_clean','norm_noise','noise','mask', '-v7.3')
