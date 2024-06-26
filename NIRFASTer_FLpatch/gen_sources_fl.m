function qvec = gen_sources_fl(mesh,frequency,data)
% GEN_SOURCES_FL Calculates FEM sources vector for re-emission.
% 
% [QVEC] = GEN_SOURCES(MESH,OMEGA,DATA) MESH is NIRFAST mesh structure.
%   FREQUENCY is modulation frequency. DATA is FEM data of excitation.
%   QVEC is a sparse, complex matrix of size N by M of initial photon
%   fluence rate at N nodes for M sources. Spatially integrated photon
%   fluence rate for each source equals 1 + 1j*eps. 
%
%   See also GEN_SOURCES, GEN_SOURCE_FL 
% 
%% check in/out

narginchk(3,3);
nargoutchk(0,1);

%% BODY
[index_num] = sources_active_index(mesh);

% nnodes = length(mesh.nodes);
nsource = numel(index_num);

% The size will match the emission qvec in case of having inactive sources
% qvec = zeros(nnodes,nsource);
% Simplify the RHS of emission equation
omega = 2*pi*frequency*1e6;
beta = mesh.gamma.*(1-(1i.*omega.*mesh.tau));
% get rid of any zeros!
if frequency == 0
    beta(beta==0) = 1e-20;
else
    beta(beta==0) = complex(1e-20,1e-20);
end

% if isCUDA
%    qvec = gen_source_fl_CUDA(mesh.nodes, mesh.elements, repmat(beta, 1, nsource).*data.phix);
% elseif isCL
%     qvec = gen_source_fl_CL(mesh.nodes, mesh.elements, repmat(beta, 1, nsource).*data.phix);
%else
    qvec = gen_source_fl_CPU(mesh.nodes, mesh.elements, repmat(beta, 1, nsource).*data.phix);
%end

%% Catch error in source vector
qvec_mask = sum(qvec,1) ~= 0; % mask of good sources
if sum(qvec_mask) ~= nsource % if not all sources are good
    index_num = reshape(index_num,1,length(index_num)); % prevents from message composition error.
    warning(['Potentially inaccurate fluorescence of sources: ' num2str(index_num(~qvec_mask))]);
end
