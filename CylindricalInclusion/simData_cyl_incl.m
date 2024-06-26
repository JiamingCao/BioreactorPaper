addpath(genpath('../../NIRFASTer'))
clear

mesh = load_mesh('cylinder_fl_full3');

%%
samples = 250;
mesh.muaf = zeros(size(mesh.muaf)); % no background fluorescence
% set the background optical properties
% mesh.muax = 1e-3*ones(size(mesh.muax));
mesh.muax = 5e-4*ones(size(mesh.muax));
mesh.muam = mesh.muax;
% mesh.musx = 0.2*ones(size(mesh.musx));
mesh.musx = 0.1*ones(size(mesh.musx));
mesh.musm = mesh.musx;
mesh.kappax = 1./(3*(mesh.muax + mesh.musx));
mesh.kappam = 1./(3*(mesh.muam + mesh.musm));

num_nodes = size(mesh.nodes, 1);
max_blobs = 3;
blob_r_rng = [5,10];    % mm
blob_h_rng = [10, 30]; % mm
% blob_muaf_rng = [3,7];   % times baseline
blob_muaf_rng = [1e-3,1e-1];   % mm-1
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
all_r = zeros(max_blobs, samples);
all_h = zeros(max_blobs, samples);
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
    blob_h = rand(num_blob,1) * (blob_h_rng(2) - blob_h_rng(1)) + blob_h_rng(1);
    blob_muaf = rand(num_blob,1) * (blob_muaf_rng(2) - blob_muaf_rng(1)) + blob_muaf_rng(1);

    idx_blobs = zeros(num_nodes, max_blobs);
    while 1
        blob_x = rand(num_blob,1) * (boundary(2)-boundary(1)) + boundary(1);
        blob_y = rand(num_blob,1) * (boundary(4)-boundary(3)) + boundary(3);
        blob_z = rand(num_blob,1) .* (boundary(6)-boundary(5)-blob_h) + (boundary(5)+blob_h/2);

        if(any(sqrt(blob_x(1:num_blob).^2 + blob_y(1:num_blob).^2)) > 65-blob_r(1:num_blob))
            continue;
        end

        for i=1:num_blob
            idx_blobs(:,i) = vecnorm(mesh.nodes(:,1:2) - [blob_x(i),blob_y(i)], 2, 2)<blob_r(i) & mesh.nodes(:,3)<blob_z(i)+blob_h(i)/2 & mesh.nodes(:,3)>blob_z(i)-blob_h(i)/2;
        end
        if ~any(prod(idx_blobs(:,1:num_blob))) && all(sum(idx_blobs(:,1:num_blob),1))
            break;
        end
    end
    
    fluctuate = 0.3*(rand(2,1)-0.5); % fluctuate the background mua by +/-30%
    mesh2.muax = mesh.muax*(1+fluctuate(1));
    mesh2.muam = mesh2.muax;
    mesh2.musx = mesh.musx*(1+fluctuate(2));
    mesh2.musm = mesh2.musx;

    for i=1:num_blob
        mesh2.muaf(idx_blobs(:,i)>0) = blob_muaf(i);
        mesh2.muax(idx_blobs(:,i)>0) = mesh2.muax(idx_blobs(:,i)>0) + blob_muaf(i);
    end
    
    mesh2.kappax = 1./(3*(mesh2.muax + mesh2.musm));
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
    all_r(1:num_blob,rep) = blob_r;
    all_h(1:num_blob,rep) = blob_h;
    all_fluctuate(:,rep) = fluctuate;
end

%% Prepare for NN
xgrid = linspace(-65,65,48);
ygrid = linspace(-65,65,48);
zgrid = linspace(0,150,56);
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);

all_muaf2 = mesh.vol.mesh2grid*all_muaf;
clean_img = reshape(all_muaf2,48,48,56,samples);

[J, data0]= jacobiangrid_fl(mesh,[],[],[],0,solver, opt);
J = rmfield(J, 'completem');

recon = zeros(size(J.complexm,2), samples);
for i=1:samples
    idx = all_datafl(:,i)>0.8*std(all_datafl(:,i)) & all_datax(:,i)>0.8*std(all_datax(:,i));
    recon(:,i) = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx),1,all_datafl(idx,i)./all_datax(idx,i));
end
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

save('images_reactor_cyl_H15', 'clean_img', 'noisy_img', 'inmesh','all_x', 'all_y', 'all_z', 'all_r', 'all_h', 'all_nblob', 'all_muaf', 'all_datafl', 'all_datax', 'all_noise', 'all_fluctuate', 'all_datax_clean', 'all_datafl_clean', 'mask', '-v7.3')
