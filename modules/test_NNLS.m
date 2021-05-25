function varargout = test_NNLS( what, varargin )
switch (what) 
    case 'simulate_IID' 
        N=8; 
        P1 = 6;
        P2 = 5; 
        vararginoptions(varargin,{'N','P1','P2'}); 
        X= normrnd(0,1,N,P1); 
        X= bsxfun(@minus,X,mean(X,1)); 
        X= bsxfun(@rdivide,X,sqrt(sum(X.^2,1)/size(X,1))); 
        
        W = normrnd(0,1,P1,P2); 
        W(W<0)=0; 
        
        Y = X * W + normrnd(0,1,N,P2); 
        varargout={X,Y}; 
    case 'NNLS_speed_test' 
        T.P1 = [10;20;30;40;50;70;100;200;300;400]; 
        lambda = [0.01,0]; 
        for i=1:length(T.P1) 
            [X,Y]=test_NNLS('simulate_IID','N',42,'P1',T.P1(i),'P2',100); 
    
            tic; 
            [N,P]= size(Y);
            [N,Q]= size(X);
            XX=X'*X;
            XY=X'*Y;
            A = -eye(Q);
            b = zeros(Q,1);
            u=nan(Q,P);  % Make non-calculated to nan to keep track of missing voxels
            for p=find(~isnan(sum(Y)))
                u(:,p) = cplexqp(XX+lambda(2)*eye(Q),ones(Q,1)*lambda(1)-XY(:,p),A,b);
            end;
            T.time(i,1)=toc; 
        end
        varargout={T}; 
end 