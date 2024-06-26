addpath(genpath('/home/jcao/Documents/NIRFASTer'))
clear

mesh = load_mesh('cylinder_equal');

%%
samples = 2500;
num_nodes = size(mesh.nodes, 1);
max_blobs = 2;
blob_r_rng = [7,10];    % mm
blob_muaf_rng = [3,5];   % times baseline
boundary = [min(mesh.nodes(:,1)), max(mesh.nodes(:,1)), min(mesh.nodes(:,2)), max(mesh.nodes(:,2)), min(mesh.nodes(:,3)), max(mesh.nodes(:,3))];
radius = boundary(2);

all_muaf = zeros(size(mesh.nodes,1), samples);
all_datax = zeros(size(mesh.link,1), samples);
all_datafl = zeros(size(mesh.link,1), samples);

all_x = zeros(max_blobs, samples);
all_y = zeros(max_blobs, samples);
all_z = zeros(max_blobs, samples);
all_nblob = zeros(samples,1);

solver=get_solver('BiCGStab_CPU');

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
    
    for i=1:num_blob
        blob=[];
        blob.x = blob_x(i);
        blob.y = blob_y(i);
        blob.z = blob_z(i);
        blob.r = blob_r(i);
        blob.muaf = blob_muaf(i)*mesh.muaf(1);
        mesh2 = add_blob(mesh2, blob);
    end
    mesh2.muax = mesh.muax;

    data = femdata_fl(mesh2, 0, solver);
    all_muaf(:,rep) = mesh2.muaf;
    all_datafl(:,rep) = data.amplitudefl + 0.01*data.amplitudefl.*randn(size(data.amplitudefl));
    all_datax(:,rep) = data.amplitudex + 0.01*data.amplitudex.*randn(size(data.amplitudex));
    all_nblob(rep) = num_blob;
    all_x(1:num_blob,rep) = blob_x;
    all_y(1:num_blob,rep) = blob_y;
    all_z(1:num_blob,rep) = blob_z;
end

%% Prepare for NN
xgrid = linspace(-32,32,32);
ygrid = linspace(-32,32,32);
zgrid = linspace(-32,32,32);
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);

all_muaf2 = mesh.vol.mesh2grid*all_muaf;
clean_img = reshape(all_muaf2,32,32,32,samples);

[J, data]= jacobiangrid_fl(mesh,[],[],[],0,solver);
J = rmfield(J, 'completem');
[~, invop] = tikhonov(J.complexm./data.amplitudex,10);
recon = invop*(all_datafl./all_datax);
noisy_img = reshape(recon,32,32,32,samples);

inmesh = mesh.vol.gridinmesh;

for i=1:2500
    tmp=clean_img(:,:,:,i);
    clean_img(:,:,:,i)=tmp/std(tmp(:));
end
for i=1:2500
    tmp=noisy_img(:,:,:,i);
    noisy_img(:,:,:,i)=tmp/std(tmp(:));
end

try
    save('images', 'clean_img', 'noisy_img', 'inmesh','all_x', 'all_y', 'all_z', 'all_nblob', 'all_muaf', 'all_datafl', 'all_datax')
catch
    save('images', 'clean_img', 'noisy_img', 'inmesh','all_x', 'all_y', 'all_z', 'all_nblob', 'all_muaf', 'all_datafl', 'all_datax','-v7.3')
end
