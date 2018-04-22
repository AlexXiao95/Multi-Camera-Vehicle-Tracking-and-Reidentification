function model = trainBayesian2(feat, training_labels, varargin)

%training label is a n*1 matrix containing the label of each training
%sample.

epochs = 5;
if nargin > 2
    epochs = varargin{1};
end

is_plot = true;
if nargin > 3
    is_plot = varargin{2};
end

disp('Preprocessing...');
labels = unique(training_labels)';

counts = 1;
for label = labels
    person_id2img_id{counts} = find(training_labels==label);
    counts = counts+1;
end

% person_id2img_id=[]
n = size(feat,1);
person_n = length( person_id2img_id );

dim = size( feat, 2);
mus = zeros( person_n , dim );

sigma_e = zeros(dim, dim );
simga_e_norm = 0;

max_m = 0;
es = zeros( size(feat) );

disp('Initializing...');
for j = 1 : person_n
    temp_feat = feat(person_id2img_id{j},:);    
    mu = mean(temp_feat);
    mus(j,:) = mu;
    
    if 0
        temp_sigma_e = zeros( dim , dim );
        for k = 1:size( temp_feat, 1)
            diff = temp_feat( k ,: ) - mu;
            temp_sigma_e = temp_sigma_e + diff'*diff;
        end
        temp_sigma_e = temp_sigma_e / size( temp_feat, 1);
		if is_plot
			subplot(121); imagesc( temp_sigma_e )
			subplot(122); imagesc( cov(temp_feat) )
		end
    else
        m = size( temp_feat, 1);
        max_m = max( max_m , m );
        es(person_id2img_id{j},:) = temp_feat - repmat( mu, m , 1);
        mj(j) = m;
    end    
end

sigma_mu = cov( mus ) + eye(dim,dim)*1e-5;
sigma_e = cov(es) * (size(es,1)-1) / (size(es,1)-size(mus,1))+ eye(dim,dim)*1e-5;

%% run EM
disp('Starting EM...');
eff_m = unique(mj);

eff_m(eff_m==0)=[];

if 1
    for EM_times = 1:epochs
        %E Step
        fprintf(' E_step ')
        nsigma_e = zeros(size(sigma_e ) );
        nsimga_e_norm = 0;

        F = inv(sigma_e);
        
        clear G
        %compute effective m is enough
        for m = eff_m
            G{m} = - inv( m * sigma_mu + sigma_e ) * sigma_mu * F;

            
        end
        es = zeros( size(feat) );                

        for j = 1 : person_n
            if(mod(j,200)==0), fprintf('.');end
            temp_feat = feat(person_id2img_id{j},:);

            m = size( temp_feat, 1);
            
            mus(j,:) = zeros( 1, dim );
            
%             for k = 1:m
%                 mus(j,:) = mus(j,:) + (sigma_mu * ( F + G{m} * m) * temp_feat(k,:)')';
%             end
            
%       modification by XIONG
%       use matrix multipication is much faster than for loops
            mus(j,:) = sum(sigma_mu*(F+G{m}*m)*temp_feat',2)';

%       modification by XIONG
%       same as previous 
            inter = sum(sigma_e*G{m}*temp_feat',2)';
            es(person_id2img_id{j},:) = bsxfun(@plus, temp_feat, inter);
            
        end

        fprintf(' M_step ')
        sigma_mu = cov( mus ) + eye(dim,dim)*1e-5;
        %real_freedom = size(es,1)-size(mus,1);
        sigma_e = cov( es ) *(size(es,1)-1) / (size(es,1)-size(mus,1)) + eye(dim,dim)*1e-5;
        %sigma_e = cov( es )  + eye(dim,dim)*1e-5;
        fprintf(' done\n')
		if is_plot
			subplot(121)
			imagesc( sigma_mu+sigma_e);
			subplot(122)
			imagesc(sigma_mu);
			drawnow;
		end
    end
end

%% formating output

sigma_e = sigma_mu + sigma_e;

model.HE = [ sigma_e zeros(dim,dim); zeros(dim,dim) sigma_e ];
model.HI = [ sigma_e sigma_mu; sigma_mu sigma_e ];
model.MU = mean( mus);
end

