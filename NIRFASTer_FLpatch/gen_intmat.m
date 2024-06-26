function [mesh, int_mat, int_mat2] = gen_intmat(mesh, xgrid, ygrid, zgrid)
% Generates an integration matrix that converts mesh data to a regular grid
% This will add an extra field "vol" to the mesh, which stores the
% integration matrice to and from the grid as well as the grid itself
%
% Input: 
% mesh: NIRFAST mesh
% xgrid, ygrid, zgrid: 3D coordinates of the grid to be interpolated on
% For 2D meshes, leave zgrid empty
%
% Output:
% mesh: the input mesh with appended field "vol"
% int_mat: integration matrix from mesh data to volumetric data
% int_mat2: integration matrix from volumetric data back to mesh data
%
% Jiaming Cao, 2023

if nargin == 3
    zgrid = [];
end
if isempty(zgrid)
    [xgrid1, ygrid1] = meshgrid(xgrid, ygrid);
    coord = [xgrid1(:), ygrid1(:)];
else
    [xgrid1, ygrid1, zgrid1] = meshgrid(xgrid, ygrid, zgrid);
    coord = [xgrid1(:), ygrid1(:), zgrid1(:)];
end
[ind, int_func] = mytsearchn_fast(mesh, coord);
% [ind, int_func] = tsearchn(mesh.nodes, mesh.elements, coord);

inside = find(~isnan(ind));
nodes = mesh.elements(ind(inside),:);
int_func_inside = int_func(inside, :);
int_mat = sparse(repmat(inside, size(int_func,2), 1), nodes(:), int_func_inside(:), length(ind), size(mesh.nodes,1));

vol = [];
vol.xgrid = xgrid;
vol.ygrid = ygrid;
vol.zgrid = zgrid;
vol.mesh2grid = int_mat;
vol.gridinmesh = inside;
if ~isempty(zgrid)
    vol.res = [xgrid(2)-xgrid(1), ygrid(2)-ygrid(1), zgrid(2)-zgrid(1)];
else
    vol.res = [xgrid(2)-xgrid(1), ygrid(2)-ygrid(1)];
end

gridDT = delaunayTriangulation(coord);
mesh2.elements = gridDT.ConnectivityList;
mesh2.nodes = gridDT.Points;
mesh2.dimension = mesh.dimension;

[ind, int_func] = mytsearchn_fast(mesh2, mesh.nodes);

% Now calculate the matrix to map the data from grid back to mesh
inside = find(~isnan(ind));
% if any of the queried nodes was not asigned a value in the previous step,
% treat it as an outside node and extrapolate. Otherwise the boundary
% elements will have smaller values than they should
tmp = mesh2.elements(ind(inside), :);
tmp2 = reshape(ismember(tmp(:),vol.gridinmesh), length(inside),[]);
outside = inside(sum(tmp2,2) < size(tmp2, 2));

outside = sort([find(isnan(ind)); outside]);
inside = setdiff(inside, outside);

gridDT = delaunayTriangulation(mesh2.nodes(vol.gridinmesh, :));
% [nn] = dsearchn(mesh2.nodes(vol.gridinmesh, :), mesh.nodes(outside, :));
nn = dsearchn(gridDT.Points, gridDT.ConnectivityList, mesh.nodes(outside, :));

nodes = mesh2.elements(ind(inside),:);
int_func_inside = int_func(inside, :);
% int_mat2 = sparse([repmat(inside, size(int_func,2), 1)], [nodes(:)], [int_func_inside(:)], length(ind), size(mesh2.nodes,1));
int_mat2 = sparse([repmat(inside, size(int_func,2), 1); outside], [nodes(:); vol.gridinmesh(nn)], [int_func_inside(:); ones(length(nn),1)], length(ind), size(mesh2.nodes,1));
vol.grid2mesh = int_mat2;
vol.meshingrid = inside;

mesh.vol = vol;
