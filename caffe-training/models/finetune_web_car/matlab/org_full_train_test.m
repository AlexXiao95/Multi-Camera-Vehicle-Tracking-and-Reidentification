clear,close all,clc
%new web-nature data statistics
%in a 3-level tree
load select_data_m
load model_brand_idx
train_r = 0.7;
test_r = 1-train_r;

dataDir='/home/ljyang/work/data/CompCars/image/';
labDir = '/home/ljyang/work/data/CompCars/label/';
imDir= dataDir;
savDir='../lists/';
rng(0);
load(['classification_im_n']);
% im_n = zeros(length(s_id),1);
types = {'classification_view','verification','verification_add'};
for t=1:1
saveDir=['D:\car_dataset\Xcar_org\',types{t},'\'];%use s_id
% saveDir='D:\car_dataset\Xcar_org\verification\';%use s_v_id
% saveDir='D:\car_dataset\Xcar_org\verification_add\'; %use v_a_id
%s_id = v_a_id;
if t==2
    s_id = s_v_id;
elseif t==3
    s_id = v_a_id;
end
model_n = length(s_id);

train_filename = [savDir,types{t},'_train'];
test_filename = [savDir,types{t},'_test'];
f_train = fopen(train_filename,'w');
f_test = fopen(test_filename,'w');
min_s=80;

for i=1:length(s_id)
        model_id = s_id(i);
        make_id = model_brand_idx(model_id);
        year = dir([imDir,num2str(make_id),'/',num2str(model_id)]);
        year = year(3:end);
        train_n = round(im_n(i)*train_r);
        dup_time = max(1,round(min_s/train_n));
        p =randperm(im_n(i));
        c=0;
        for k=1:length(year)
            list = dir([imDir,num2str(make_id),'/',num2str(model_id),'/',year(k).name,'/*.jpg']);
%             im_n(i) = im_n(i)+length(list);
            for j=1:length(list)
                c=c+1;
                %read label file
                label = textread([labDir,num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name(1:end-4),'.txt'],'%d');
                v = max(-1,label(1)-1); 
%                 if (label(1)==0) 
%                     error('view annotation is 0');
%                 end
                %random split to train and test
                
                if p(c) <= train_n
                    for d=1:dup_time
                      fprintf(f_train,'%s%s %d %d\n',dataDir,[num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name],i-1, v);
                    end
                    
%                     train_n(i) = train_n(i)+1;
                else
                    fprintf(f_test,'%s%s %d %d\n',dataDir,[num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name],i-1, v);
             
%                     test_n(i) = test_n(i)+1;
                end
            end

           

            
        end
    
end
fclose(f_train);
fclose(f_test);
end
% save(['classification_im_n'], 'im_n');