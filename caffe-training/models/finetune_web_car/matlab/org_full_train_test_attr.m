clear,close all,clc
%new web-nature data statistics
%in a 3-level tree
load select_data_m
load attr_m%s_attr,s_v_attr
load model_brand_idx
%regulize attr
mean1 = mean(s_attr(:,1));
std1 = std(s_attr(:,1));
mean2 = mean(s_attr(:,2));
std2 = std(s_attr(:,2));
% save attr_reg mean1 std1 mean2 std2
s_attr(:,1) = (s_attr(:,1)-mean1)/std1;
s_attr(:,2) = (s_attr(:,2)-mean2)/std2;

s_v_attr(:,1) =(s_v_attr(:,1)-mean1)/std1;
s_v_attr(:,2) = (s_v_attr(:,2)-mean2)/std2;


dataDir='/home/ljyang/work/data/CompCars/image/';
imDir= dataDir;
savDir='../lists/';

for t=1:2

if t==2
    s_id = s_v_id;
elseif t==3
    s_id = v_a_id;
end
model_n = length(s_id);
if t==1
    state = 'train';
    attr = s_attr;
else
    state = 'test';
    attr = s_v_attr;
end
filename = [savDir,state,'_attr_disc'];
filename2 = [savDir,state,'_attr_list'];

f = fopen(filename,'w');
f2 = fopen(filename2,'w');

for i=1:length(s_id)
        model_id = s_id(i);
        make_id = model_brand_idx(model_id);
        year = dir([imDir,num2str(make_id),'/',num2str(model_id)]);
        year = year(3:end);
        for k=1:length(year)
            list = dir([imDir,num2str(make_id),'/',num2str(model_id),'/',year(k).name,'/*.jpg']);
%             im_n(i) = im_n(i)+length(list);
            for j=1:length(list)
              
                fprintf(f,'%s%s %d %d %d\n',dataDir,[num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name],attr(i,3),attr(i,4),attr(i,5));
                    
                fprintf(f2,'%s\n',[num2str(make_id),'/',num2str(model_id),'/',year(k).name,...
                        '/',list(j).name]);
%                     train_n(i) = train_n(i)+1;
                
            end

           

            
        end
    
end
fclose(f);
fclose(f2);
end
% save([savDir,'classification_im_n'], 'im_n');