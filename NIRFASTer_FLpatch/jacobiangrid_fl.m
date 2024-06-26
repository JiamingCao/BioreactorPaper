function [J,data_grid, sys_x, sys_m]=jacobiangrid_fl(mesh,xgrid, ygrid, zgrid,varargin)

%% check in/out

narginchk(1,8);
nargoutchk(0,4);

%% Use grid information stored in mesh if not specified
if nargin == 1
    xgrid = [];
    ygrid = [];
    zgrid = [];
end

%% get optional inputs, handle variable input

% default
frequency = 0;
solver = get_solver;
OPTIONS = solver_options;

if ~isempty(varargin)
    if length(varargin) >= 1
        % frequency
        if ~ischar(varargin{1}) && ~isstring(varargin{1}) && (numel(varargin{1})==1)
            % sanity check
            if varargin{1} < 0
                error('Negative frequency value. Please see help for details on how to use this function.')
            else
                frequency = varargin{1};
            end
        else
            error('Bad 1st argument value. A scalar expected. Please see help for details on how to use this function.')
        end
    end
    if length(varargin) >= 2
        % solver
        if ischar(varargin{2}) || isstring(varargin{2})
            % user specified, sanity check
            solver = get_solver(varargin{2});
        elseif isstruct(varargin{2})
            OPTIONS = varargin{2};
        elseif isempty(varargin{2})
            solver = get_solver;
        else
            error('Bad 2nd argument value. Solver name or solver settings structure expected. Please see help for details on how to use this function.')
        end
    end
    if length(varargin) >= 3
        % solver options
        if isstruct(varargin{3})
            OPTIONS = varargin{3};
        elseif isempty(varargin{3})
            OPTIONS = solver_options;
        else
            error('Bad 3rd argument value. Solver settings structure expected. Please see help for details on how to use this function.')
        end
    end
end

%% If not a workspace variable, load mesh
if ~isstruct(mesh)
    mesh = load_mesh(mesh);
end

%% Calculate field and boundary data for all sources
tmp_mesh = mesh;
tmp_mesh.mm = 0;
[data, ~,~,~,~, sys_x, sys_m] = femdata_fl(tmp_mesh, frequency, solver, OPTIONS);
if isempty(xgrid) && isempty(ygrid) && isempty(zgrid)
    if isfield(mesh, 'vol')
        data_grid = data2grid(data, mesh);
        xgrid = mesh.vol.xgrid;
        ygrid = mesh.vol.ygrid;
        zgrid = mesh.vol.zgrid;
    else
        error('Please supply grid information.')
    end
else
    [data_grid, mesh] = data2grid(data, mesh, xgrid, ygrid, zgrid);
end

%% Now calculate adjoint
% make sources copy and swap with detectors
tmp_mesh = mesh;
tmp_mesh.phix = 0;
tmp_mesh.fl = 0;
tmp_mesh.source = [];
tmp_mesh.meas = [];
% swap source
tmp_mesh.source.fixed = 0;
tmp_mesh.source.num = mesh.meas.num;
tmp_mesh.source.coord = mesh.meas.coord;
tmp_mesh.source.fwhm = zeros(size(mesh.meas.num));
tmp_mesh.source.int_func = mesh.meas.int_func;
%swap detector
tmp_mesh.meas.fixed = 0;
tmp_mesh.meas.num = mesh.source.num;
tmp_mesh.meas.coord = mesh.source.coord;
%swap link
tmp_mesh.link(:,1) = mesh.link(:,2);
tmp_mesh.link(:,2) = mesh.link(:,1);
% a missing detectors 'mesh.meas.int_func' will calculate itself in 'get_boundary_data' if needed

% calculate adjoint field for all sources (detectors)
data_detector = femdata_fl(tmp_mesh, frequency, solver, OPTIONS);
data_grid2 = data2grid(data_detector, tmp_mesh);
data_grid.aphim = data_grid2.phimm;
%% Build Jacobian

active_idx = mesh.link(:,3)==1;
link = mesh.link(active_idx,:);

if xor(isreal(data_grid.phix), isreal(data_grid.aphim))
    error('Both direct and adjoint fields should be real or complex at the same time.');
else
    phi = reshape(data_grid.phix, [], size(data_grid.phix, mesh.dimension+1));
    aphi = reshape(data_grid.aphim, [], size(data_grid.aphim, mesh.dimension+1));
    J.complexm = IntGrid_CPU(phi, aphi, link);
    % J.complexm = tmp1(:, mesh.link(:,1)) .* tmp2(:, mesh.link(:,2));
    J.complexm = J.complexm' * prod(mesh.vol.res);
    J.completem = J.complexm./data_grid.complexfl(active_idx);
end


