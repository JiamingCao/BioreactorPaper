function [data,mesh,varargout]=femdata_fl(mesh,frequency,varargin)

% [data,mesh]=femdata_FL(mesh,frequency)
%
% Calculates fluorescence data (phase and amplitude) for a given
% mesh at a given frequency (MHz).
% outputs phase and amplitude in structure data
% and mesh information in mesh

% From NIRFAST9.1 with minor modification
% Jiaming Cao, 2023

%% get optional inputs, handle variable input

% default
solver = get_solver;
OPTIONS = solver_options;

if ~isempty(varargin)
    if length(varargin) == 1
        if ischar(varargin{1}) || isstring(varargin{1})
            % user specified, sanity check
            solver = get_solver(varargin{1});
        elseif isstruct(varargin{1})
            OPTIONS = varargin{1};
        else
            error('Bad 3rd argument value. Text or structure expected. Please see help for details on how to use this function.')
        end
    elseif length(varargin) == 2
        
        if ischar(varargin{1}) || isstring(varargin{1})
            % user specified, sanity check
            solver = get_solver(varargin{1});
        else
            error('Bad 3rd argument value. Text expected. Please see help for details on how to use this function.')
        end
        
        if isstruct(varargin{2})
            OPTIONS = varargin{2};
        else
            error('Bad 4th argument value. Structure expected. Please see help for details on how to use this function.')
        end
    else
        error('Bad arguments. Please see the help for details on how to use this function.')
    end
end

%% If not a workspace variable, load mesh
if ischar(mesh)== 1
    mesh = load_mesh(mesh);
end
if ~strcmp(mesh.type,'fluor')
    warning('NIRFAST:warning:meshType',['Mesh type is ''' mesh.type '''. ''fluor'' expected. This might give unexpected results.'])
end

% Use fields to flag what forward models to calculate
if isfield(mesh,'phix') == 1
    data.phix = mesh.phix;
    xflag = 0;  % Excitation field phix is a field in the mesh
    % structured variable, so skip this calculation.
else
    xflag = 1;
end

% Use fields to flag what forward models to calculate
if isfield(mesh,'mm') == 0
    mmflag = 1; % Run forward model for intrinsic emission field
elseif isfield(mesh,'mm') == 1
    if mesh.mm == 1
        mmflag = 1; % Run forward model for intrinsic emission field
    elseif mesh.mm ~= 1
        mmflag = 0; % Skip forward model for intrinsic emission field
    end
end

if isfield(mesh,'fl') == 0
    flflag = 1; % Run forward model for fluorescence emission field
elseif isfield(mesh,'fl') == 1
    if mesh.fl == 1
        flflag = 1; % Run forward model for fluorescence emission field
    elseif mesh.fl ~= 1
        flflag = 0; % Skip forward model for fluorescence emission field
        % (used to calculate excitation field only in recons)
    end
end

% error checking
if frequency < 0
    errordlg('Frequency must be nonnegative','NIRFAST Error');
    error('Frequency must be nonnegative');
end

% modulation frequency
omega = 2*pi*frequency*1e6;

% set fluorescence variables
mesh.gamma = (mesh.eta.*mesh.muaf)./(1+(omega.*mesh.tau).^2);

%% Create FEM matrices
if xflag == 1
    % Excitation FEM matrix
    dummy_mesh = [];
    dummy_mesh.nodes = mesh.nodes;
    dummy_mesh.elements = mesh.elements;
    dummy_mesh.bndvtx = mesh.bndvtx;
    dummy_mesh.mua = mesh.muax;
    dummy_mesh.kappa = mesh.kappax;
    dummy_mesh.ksi = mesh.ksi;
    dummy_mesh.c = mesh.c;
    if isCUDA
        if isfield(OPTIONS,'GPU')
                [i_index_x, j_index_x, value_x] = gen_mass_matrix_FD_CUDA(dummy_mesh,frequency,OPTIONS.GPU);
        else
                [i_index_x, j_index_x, value_x] = gen_mass_matrix_FD_CUDA(dummy_mesh,frequency);
        end
    else
        [i_index_x, j_index_x, value_x] = gen_mass_matrix_FD_CPU(dummy_mesh,frequency);
    end
    if nargout > 5
        sys_x = [];
        sys_x.i_index_x = i_index_x;
        sys_x.j_index_x = j_index_x;
        sys_x.value_x = value_x;
        varargout{4} = sys_x;
    end

end

if mmflag == 1 || flflag == 1
    % Emission FEM matrixmuaf
    dummy_mesh = [];
    dummy_mesh.nodes = mesh.nodes;
    dummy_mesh.elements = mesh.elements;
    dummy_mesh.bndvtx = mesh.bndvtx;
    dummy_mesh.ksi = mesh.ksi;
    dummy_mesh.c = mesh.c;
    dummy_mesh.mua = mesh.muam;
    dummy_mesh.kappa = mesh.kappam;
    if isCUDA
        if isfield(OPTIONS,'GPU')
                [i_index_m, j_index_m, value_m] = gen_mass_matrix_FD_CUDA(dummy_mesh,frequency,OPTIONS.GPU);
        else
                [i_index_m, j_index_m, value_m] = gen_mass_matrix_FD_CUDA(dummy_mesh,frequency);
        end
    else
        [i_index_m, j_index_m, value_m] = gen_mass_matrix_FD_CPU(dummy_mesh,frequency);
    end
    if nargout > 6
        sys_m = [];
        sys_m.i_index_m = i_index_m;
        sys_m.j_index_m = j_index_m;
        sys_m.value_m = value_m;
        varargout{5} = sys_m;
    end

end

%% Now calculate excitation source vector
if xflag == 1 || mmflag == 1
    qvec = gen_sources(mesh);
    % Catch zero frequency (CW) here
    if frequency == 0
        qvec = abs(qvec);
    end
end

%% Calculate INTRINSIC FIELDS

% Calculate INTRINSIC EXCITATION field for all sources
if xflag == 1
    % OPTIONS.no_of_iter = 1000;% (default 1000)
    % OPTIONS.tolerance = 1e-8;% (default 1e-12 for DOUBLE and 1e-8 for SINGLE);
    % % below optins ignored for MATLAB iterative method
    % OPTIONS.rel_tolerance = 1e-8;% (default 1e-12 for DOUBLE and 1e-8 for SINGLE);
    % OPTIONS.divergence_tol = 1e8;% (default 1e8 for DOUBLE and SINGLE);

    % get photon fluence rate
    % if GPU in use
    if strcmp(solver,solver_name_GPU)
        % check if we need to return the info structureas well
        if nargout >= 3
            [data.phix, varargout{1}] = get_field_FD_CUDA(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        else
            data.phix = get_field_FD_CUDA(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        end
    % if no GPU in use, use CPU
    elseif strcmp(solver,solver_name_CPU)
        if nargout >= 3
            [data.phix, varargout{1}] = get_field_FD_CPU(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        else
            data.phix = get_field_FD_CPU(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        end
    elseif strcmp(solver,solver_name_matlab_iterative)
        if nargout >= 3
            [data.phix, varargout{1}] = get_field_FD_bicgstab_matlab(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        else
            data.phix = get_field_FD_bicgstab_matlab(i_index_x, j_index_x, value_x, qvec, OPTIONS);
        end
    else
        % use MATLAB backslash
        % +1 as the i_index and j_index are zero-based and MATLAB uses 1-based indexing
        data.phix = full(sparse(i_index_x+1, j_index_x+1, value_x)\qvec);
        if nargout >= 3
            varargout{1} = [];
        end
    end
end

% Calculate INTRINSIC field at EMISSION WAVELENGTH laser source
if mmflag == 1
    if strcmp(solver,solver_name_GPU)
        % check if we need to return the info structureas well
        if nargout >= 4
            [data.phimm, varargout{2}] = get_field_FD_CUDA(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phimm = get_field_FD_CUDA(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    % if no GPU in use, use CPU
    elseif strcmp(solver,solver_name_CPU)
        if nargout >= 4
            [data.phimm, varargout{2}] = get_field_FD_CPU(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phimm = get_field_FD_CPU(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    elseif strcmp(solver,solver_name_matlab_iterative)
        if nargout >= 4
            [data.phimm, varargout{2}] = get_field_FD_bicgstab_matlab(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phimm = get_field_FD_bicgstab_matlab(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    else
        % use MATLAB backslash
        % +1 as the i_index and j_index are zero-based and MATLAB uses 1-based indexing
        data.phimm = full(sparse(i_index_m+1, j_index_m+1, value_m)\qvec);
        if nargout >= 4
            varargout{2} = [];
        end
    end
end

clear qvec;
%% FLUORESCENCE EMISSION FIELDS
if flflag == 1
    qvec = gen_sources_fl(mesh, omega, data);
    % Catch zero frequency (CW) here
    if frequency == 0
        qvec = abs(qvec);
    end
    
    % Calculate FLUORESCENCE EMISSION field for all sources
    if strcmp(solver,solver_name_GPU)
        % check if we need to return the info structureas well
        if nargout == 5
            [data.phifl, varargout{3}] = get_field_FD_CUDA(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phifl = get_field_FD_CUDA(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    % if no GPU in use, use CPU
    elseif strcmp(solver,solver_name_CPU)
        if nargout == 5
            [data.phifl, varargout{3}] = get_field_FD_CPU(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phifl = get_field_FD_CPU(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    elseif strcmp(solver,solver_name_matlab_iterative)
        if nargout == 5
            [data.phifl, varargout{3}] = get_field_FD_bicgstab_matlab(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        else
            data.phifl = get_field_FD_bicgstab_matlab(i_index_m, j_index_m, value_m, qvec, OPTIONS);
        end
    else
        % use MATLAB backslash
        % +1 as the i_index and j_index are zero-based and MATLAB uses 1-based indexing
        data.phifl = full(sparse(i_index_m+1, j_index_m+1, value_m)\qvec);
        if nargout == 5
            varargout{3} = [];
        end
    end
end

%% EXTRACT DATA
data.link = mesh.link;

if xflag == 1
    % Extract Excitation data
    [data.complexx]=get_boundary_data(mesh,data.phix);
    % Map complex data to amplitude
    data.amplitudex = abs(data.complexx);
    % Map complex data to phase
    data.phasex = atan2(imag(data.complexx),...
        real(data.complexx));
    % Calculate phase in degrees
    data.phasex(data.phasex<0) = data.phasex(data.phasex<0) + (2*pi);
    data.phasex = data.phasex*180/pi;
    % Build data format
    data.paax = [data.amplitudex data.phasex];
end


% Extract Fluorescence Emission data
if flflag == 1
    [data.complexfl]=get_boundary_data(mesh,data.phifl);
    % Map complex data to amplitude
    data.amplitudefl = abs(data.complexfl);
    % Map complex data to phase
    data.phasefl = atan2(imag(data.complexfl),...
        real(data.complexfl));
    % Calculate phase in degrees
    data.phasefl(data.phasefl<0) = data.phasefl(data.phasefl<0) + (2*pi);
    data.phasefl = data.phasefl*180/pi;
    % Build data format
    data.paafl = [data.amplitudefl data.phasefl];
    data.paaxfl = [data.amplitudex data.phasex data.amplitudefl data.phasefl];
end

% Exrtact intrinsic emssion field data
if mmflag == 1
    [data.complexmm]=get_boundary_data(mesh,data.phimm);
    % Map complex data to amplitude
    data.amplitudemm = abs(data.complexmm);
    % Map complex data to phase
    data.phasemm = atan2(imag(data.complexmm),...
        real(data.complexmm));
    % Calculate phase in degrees
    data.phasemm(data.phasemm<0) = data.phasemm(data.phasemm<0) + (2*pi);
    data.phasemm = data.phasemm*180/pi;
    % Build data format
    data.paamm = [data.amplitudemm data.phasemm];
    if flflag == 1
        data.paaxflmm = [data.amplitudex data.phasex data.amplitudefl data.phasefl data.amplitudemm data.phasemm];
    end
end
