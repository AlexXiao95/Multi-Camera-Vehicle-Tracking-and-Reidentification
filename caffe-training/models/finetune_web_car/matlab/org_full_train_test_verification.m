clear,close all,clc
%new web-nature data statistics
%in a 3-level tree

load select_data_m
load model_brand_idx

dataDir='/home/ljyang/work/data/CompCars/image/';
imDir= dataDir;
savDir='../lists/';

s_id =s_v_id;
model_n = length(s_id);

filename = [savDir,'verification_train'];


f = fopen(filename,'w');


for i=1:length(s_id)
        model_id = s_id(i);
        make_id = model_brand_idx(model_id);
        year = dir([imDir,num2str(make_id),'/',num2str(model_id)]);
        year = year(3:end);
        for k=1:length(year)
            list = dir([imDir,num2str(make_id),'/',num2str(model_id),'/',year(k).name,'/*.jpg']);
%             im_n(i) = im_n(i)+length(list);
            for j=1:length(list)
              
                fprintf(f,'%s%s %d\n',dataDir,[num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name],i-1);                   
               
                
            end
       
        end
    
end
fclose(f);


%% saving test data for verification
mode={'easy','medium','hard'};
for m=1:3
    [path1,path2,labels] = textread(['../lists/verification_pairs_',mode{m},'.txt'],'%s %s %d');
    path = [path1;path2];
    filename = [savDir,'verification_test_',mode{m}];
    f=fopen(filename,'w');
    for i=1:length(path)
        fprintf(f,'%s%s 0\n',dataDir,path{i});
    end
    fclose(f);
end
               
