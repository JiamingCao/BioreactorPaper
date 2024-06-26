function [datagrid, mesh] = data2grid(data, mesh, xgrid, ygrid, zgrid)
% Converts FEM data to grid data for easier later operations
% data: data to be converted, has the same length as number of nodes
% mesh: NIRFAST mesh
% xgrid, ygrid, zgrid (optional): 3D coordinates of the grid to be interpolated on
% If grid not provided, function uses existing integration matrix
% Jiaming Cao, 2023

silence = 1;
if nargin ==2
    if ~silence
        fprintf('Using existing integration matrix.\n');
    end
    xgrid = mesh.vol.xgrid;
    ygrid = mesh.vol.ygrid;
    zgrid = mesh.vol.zgrid;
elseif nargin == 4
    zgrid = [];
elseif ~isfield(mesh, 'vol')
    fprintf('Calculating integration matrix\n');
    mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
    fprintf('Done.\n');
else
    grid = mesh.vol;
    if length(grid.xgrid)~=length(xgrid) || length(grid.ygrid)~=length(ygrid) || length(grid.zgrid)~=length(zgrid)
        mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
        warning('Recalculating integration matrix using the new grid!')
    elseif any(grid.xgrid - xgrid) || any(grid.ygrid - ygrid) || any(grid.zgrid - zgrid)
        mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
        warning('Recalculating integration matrix using the new grid!')
    else
        if ~silence
            fprintf('Using existing integration matrix.\n');
        end
    end
end

allfields = fieldnames(data);
for i=1:length(allfields)
    if strfind(allfields{i}, 'phi')
        try
            eval(['newdata = mesh.vol.mesh2grid * data.',allfields{i},';']);
            dim = eval(['size(data.',allfields{i},',2)']);
            if mesh.dimension==3
                newdata = reshape(newdata(:), [length(ygrid), length(xgrid), length(zgrid), dim]);
            else
                newdata = reshape(newdata(:), [length(ygrid), length(xgrid), dim]);
            end
        catch
            % if the field is not present, set it to zero
            newdata = 0;
        end

        eval(['datagrid.',allfields{i},'= newdata;']);
    else
        % Copy over the other fields - they should not change
        eval(['datagrid.',allfields{i},'= data.',allfields{i},';']);
    end
end

datagrid.xgrid = xgrid;
datagrid.ygrid = ygrid;
datagrid.zgrid = zgrid;

