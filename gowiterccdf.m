function [s] = gowiterccdf(grad,Hmat,U,n,j,rows)
s = zeros(rows,1);
for k = 1:rows          
    s(k,1) = grad(k,1)/Hmat(j,j);
    s(k,1) = U{n,1}(k,j)-s(k,1);
    s(k,1) = max(s(k,1),0);
    %if ( s(k,1)< 0) 
    %   s(k,1)=0;
    %end		
end

