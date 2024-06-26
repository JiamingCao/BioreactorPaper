function [xhat, inv_op, res] = tikhonov(A, alpha, y)
% xhat = argmin_x ||Ax - y||^2 + alpha*||x||^2
m = size(A,1);
n = size(A,2);
if m > n
    inv_op = inv(A'*A + alpha*eye(n))*A';
    if exist('y') && ~isempty(y)
        xhat = inv_op * y;
    else
        xhat = nan;
    end
    if nargout == 3
        res = norm(A*xhat - y);
    end
else
    M = A/sqrt(alpha);  
    if nargout > 1
        inv_op = A' - M'*((eye(m) + M*M')\(M*A'));
        inv_op = inv_op/alpha;
    end
    if exist('y') && ~isempty(y)
        xhat = A'/alpha*y;
        xhat = xhat -  M'*((eye(m) + M*M')\(M*xhat));
        % xhat = ((eye(n) - M'*(inv(eye(m) + M*M')*M))*(A'/alpha*y));
        % xhat = inv_op * y;
    else
        xhat = nan;
    end
    if nargout == 3
        res = norm(A*xhat - y);
    end
end
