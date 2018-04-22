function sim_score = queryBayesian(model, feas_query, feas_gallery, varargin)
% Each row is a feature sample.

opt.pres_k = 100;
opt.batch_size = 8000;
opt.total_size = size(feas_query,1);

%parse additional params
if(nargin>3)
    n_prop = nargin-3;
    if mod(n_prop,2)==0
        for i =1:n_prop/2
            try
                switch(varargin{2*i-1})
                    case 'preserve'
                        opt.pres_k = varargin{2*i};
                        
                    case 'batch'
                        opt.batch_size = varargin(2*i);
                        
                    %TODO: add you custom property here
                    otherwise
                        throw(MException('ParamError:UnknownPropertyName', ...
                            'The property name "%s" is unknown',...
                            varargin{2*i-1}));
                end
            catch e
                rethrow(e);
            end
        end
    else
        error('Additional inputs must be property-value pairs.')
    end
end

%% check query and gallery size
n_query = size(feas_query, 1);
n_gallery = size(feas_gallery, 1);

%% precompute some matrix

%center the gallery samples
feas_gallery = bsxfun(@minus, feas_gallery, model.MU);

% blockwise computation, precomputed for accelration
M = inv(model.HE)- inv(model.HI);
d_feat = size(feas_query,2);
model.B1 = M(1:d_feat,d_feat+1:end);
model.B2 = M(d_feat+1:end,1:d_feat);
C = M(d_feat+1:end,d_feat+1:end);
model.gcg = sum(bsxfun(@times, feas_gallery*C, feas_gallery),2);
model.gb2 = feas_gallery*model.B2;


%% go batch operation to save memeory need for parfor
n_batch = ceil(opt.total_size/opt.batch_size);
sim_score.score = zeros(opt.total_size, opt.pres_k);
sim_score.location = zeros(opt.total_size, opt.pres_k);

tic
for i_batch = 0:n_batch-1
    ticid = tic;
    head = n_batch*i_batch+1;
    tail = min((n_batch+1)*opt.batch_size,opt.total_size);
    num = tail-head+1;
    temp=zeros(num, n_gallery);
    
    % test the qeuries against galleries
%     parfor i = 1:num
%         temp(i,:) = query_x(model,feas_query(head+i-1,:),feas_gallery);
%     end    
    
    temp = fast_query_x(model, feas_query(head:tail,:));%,feas_gallery);

    %preserve top 'pres_k' similarity scores and return them to the user
    [sim_score.score(head:tail,:), sim_score.location(head:tail,:)] = maxk(temp,opt.pres_k,2);
    toc(ticid)
end
toc
end




function [scores] = fast_query_x(model, fea_query)
    f = bsxfun(@minus,fea_query,model.MU);
    scores = bsxfun(@plus, 2*model.gb2*f',model.gcg)';

end
