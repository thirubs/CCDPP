function [U,UM] = CCDPPF(X,J,U,Y,UM,maxiters,nor,inneriters)
%%NCMTF-CCD++. Nonnegative Coupled Matrix Tensor Factorization with CCD++
% This implementation can be used for only 3-order tensor with one matrix
% coupled on mode 1.
%%
% [Input]
% X           data tensor                        (sptensor)
% Y           data matrix                        (double)
% J           Rank of the tensor and matrix
% U,UM        Randonly initialized factor matrices
% maxiters    the numer of iterations   
% nor         1 for 1norm or 2 for 2norm
% inneriters  Maximum number of inner iterations. 
% [Output]
% U           Factor matrices of the input tensor     (ktensor)
% UM          Factor matrices of the input matrix
%% Usage example
% [U,UM] = CCDPPF(X,J,U,Y,UM,maxiters,nor,inneriters)
% usage
%% Title
fprintf('*************************\n')
fprintf('NCMTF - CCD++\n');
fprintf('*************************\n')
%% Initialization for parameters 
%tic
N = ndims(X);
epsilon=1e-12;
fast = 0;
%% Error Cheacking
% None check by yourself 

%% compute Second-order derivatives for tensor factors
Hmat=sparse(ones(J,J));
for n=1:N
    Hmat= Hmat .* ((U{n,1}' *U{n,1})+epsilon);
end
Hmat2 = sparse(ones(J,J));
for n = 1:2
    Hmat2 = Hmat2.*((UM{n,1}'*UM{n,1})+epsilon);
end
%% iterations
tol = 0.001;
for i = 1:maxiters
    fprintf('%d\n',i)
    for j = 1:J % J is number of columns
	df = tol; % df is the accelarting technique that avoids frquent column updates
	count = 0;
        for q = 1: inneriters
		df2 = [];
			if (df >= tol)
				count = count +1;
				%
				for n = 1: 4 %(4 is the total number of factor matrices)
					% now update all the factor matrices
					if (n == 4)
						% Update H of input matrix's factor
						don = 2;
						rows_and_cols = size(UM{2});
						rows = rows_and_cols(1);
						Hmat2 = Hmat2 ./ ((UM{don,1}' *UM{don,1})+epsilon);
						tmphmat2 = UM{don,1}* Hmat2;
						Z = cell(2,1);
						for l = [1:don-1,don+1:2]
							Z{l} = UM{l}(:,j);
						end
						grad = double(ttv(Y, Z, -don)); % Column-wise MTTKRP calculation
						grad = -(grad-tmphmat2(:,j));
						s = gowiterccdf(grad,Hmat2,UM,don,j,rows);
						UM{don,1}(:,j)=  s;
						%Check non-negativity 
						UM{don,1}(UM{don,1}<=epsilon)=epsilon;
						if (nor > 0)
							%if (n~=N)
							   UM{don,1}=normalize_factor(UM{don,1},nor);
							%end
						end
						%Upadate Second-order derivatives with updated matrix			
						Hmat2 = Hmat2 .* ((UM{don,1}' *UM{don,1})+epsilon);
					else
						factor = n;
						rows_and_cols = size(U{n});
						rows = rows_and_cols(1);
						%cols = rows_and_cols(2);

						% Upadate Second-order derivatives for nth mode.
						Hmat = Hmat ./ ((U{n,1}' *U{n,1})+epsilon);


						%Calculate Gradient
						tmphmat = U{n,1}* Hmat;
						Z = cell(N,1);
						for l = [1:factor-1,factor+1:N]
							Z{l} = U{l}(:,j);
						end
						grad = double(ttv(X, Z, -factor)); % Column-wise MTTKRP calculation
						grad = -(grad-tmphmat(:,j));
						Hmat_a = Hmat;
						grad2 = grad; 
						s = gowiterccdf(grad2,Hmat_a,U,n,j,rows);
						if n == 1
							Hmat2 = Hmat2 ./ ((UM{n,1}' *UM{n,1})+epsilon);
							tmphmat2 = UM{n,1}* Hmat2;
							Z = cell(2,1);
							N2 = 2;
							for l = [1:n-1,n+1:N2]
								Z{l} = UM{l}(:,j);
							end
							grad2 = double(ttv(Y, Z, -n)); % Column-wise MTTKRP calculation
							grad2 = -(grad2-tmphmat2(:,j));  
							s2 = gowiterccdf(grad2,Hmat2,U,n,j,rows);
							s = s + s2;
						end 
						U{n,1}(:,j)=  s;
						if n == 1
							UM{1,1}(:,j) = s;
							%Check non-negativity 
							UM{1,1}(UM{1,1}<=epsilon)=epsilon;
							if (nor > 0)
								UM{1,1}=normalize_factor(UM{1,1},nor);
							end
						end
						%Check non-negativity 
						U{n,1}(U{n,1}<=epsilon)=epsilon;
						%Normalization (if you need)
						if(nor > 0)
							%if (n~=N)
								U{n,1}=normalize_factor(U{n,1},nor);
							%end
						end
						% Update Second-order derivatives with updated matrix			
						Hmat = Hmat .* ((U{n,1}' *U{n,1})+epsilon);
                        if (n == 1)
                            Hmat2 = Hmat2.*((U{n,1}' *U{n,1})+epsilon);
                        end
						dfi = (-(1).*s.*grad)-(0.5.*Hmat(j,j).*s.*s);
						df2(n,1) =  mean(dfi);
						
					end
					
				end  % end of n factors
				df = mean(df2);
				%q
			else
				break;
			end	 % end of df condition		
        end % end of initer for loop
    end
end

    